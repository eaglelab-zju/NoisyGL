import torch
import nni


class Logger(object):
    """
    Logger Class.

    Parameters
    ----------
    runs : int
        Total experimental runs.
    """
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result_dict):
        '''
        Add performance of a new run.

        Parameters
        ----------
        run : int
            Id of the new run.
        result_dict : dict
            A dict containing training, valid and test performances.

        '''
        assert "train" in result_dict.keys()
        assert "valid" in result_dict.keys()
        assert "test" in result_dict.keys()
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result_dict["train"])
        self.results[run].append(result_dict["valid"])
        self.results[run].append(result_dict["test"])
        self.results[run].append(result_dict['correct_labeled_train_accuracy'])
        self.results[run].append(result_dict['incorrect_labeled_train_accuracy'])
        self.results[run].append(result_dict['incorrect_labeled_mislead_train_accuracy'])
        self.results[run].append(result_dict['unlabeled_correct_supervised_accuracy'])
        self.results[run].append(result_dict['unlabeled_unsupervised_accuracy'])
        self.results[run].append(result_dict['unlabeled_incorrect_supervised_accuracy'])
        self.results[run].append(result_dict['total_time'])

    def print_statistics(self, run=None):
        '''
        Function to output the statistics.

        Parameters
        ----------
        run : int
            Id of a run. If not specified, output the statistics of all runs.

        Returns
        -------
            The statistics of a given run or all runs.

        '''
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[0]:.2f}')
            print(f'Highest Valid: {result[1]:.2f}')
            print(f'   Final Test: {result[2]:.2f}')
            return  result[2]
        else:
            best_result = 100 * torch.tensor(self.results)

            total_results = {}
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            total_results["train_accuracy"] = {"acc": r.mean(), "std": r.std}

            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            total_results["valid_accuracy"] = {"acc": r.mean(), "std": r.std()}

            r = best_result[:, 2]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            total_results["test_accuracy"] = {"acc": r.mean(), "std": r.std()}

            if nni.get_trial_id()!="STANDALONE":
                nni.report_final_result(float(r.mean()))

            r = best_result[:, 3]
            print(f'Correct Labeled Train Accuracy: {r.mean():.2f} ± {r.std():.2f}')
            total_results["correct_labeled_train_accuracy"] = {"acc": r.mean(), "std": r.std()}

            r = best_result[:, 4]
            print(f'Incorrect Labeled Train Accuracy: {r.mean():.2f} ± {r.std():.2f}')
            total_results['incorrect_labeled_train_accuracy'] = {"acc": r.mean(), "std": r.std()}

            r = best_result[:, 5]
            print(f'Incorrect Labeled Mislead Train Accuracy: {r.mean():.2f} ± {r.std():.2f}')
            total_results['incorrect_labeled_mislead_train_accuracy'] = {"acc": r.mean(), "std": r.std()}

            r = best_result[:, 6]
            print(f'Unlabeled Correct Supervised Accuracy: {r.mean():.2f} ± {r.std():.2f}')
            total_results['unlabeled_correct_supervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

            r = best_result[:, 7]
            print(f'Unlabeled Unsupervised Accuracy: {r.mean():.2f} ± {r.std():.2f}')
            total_results['unlabeled_unsupervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

            r = best_result[:, 8]
            print(f'Unlabeled Incorrect Supervised Accuracy: {r.mean():.2f} ± {r.std():.2f}')
            total_results['unlabeled_incorrect_supervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

            r = best_result[:, 9] * 0.01
            print(f'Time: {r.mean():.2f} ± {r.std():.2f}')
            total_results['total_time'] = {"mean": r.mean(), "std": r.std()}

            return total_results
