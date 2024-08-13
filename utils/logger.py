import torch
import time
import pandas as pd


class MultiExpRecorder(object):
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

    def get_statistics(self):
        '''
        Function to output the statistics.

        Parameters
        ----------
            None

        Returns
        -------
            The statistics of a given run or all runs.

        '''
        best_result = 100 * torch.tensor(self.results)
        total_results = {}
        # print(f'All runs:')
        r = best_result[:, 0]
        # print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
        total_results["train_accuracy"] = {"acc": r.mean(), "std": r.std}

        r = best_result[:, 1]
        # print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
        total_results["valid_accuracy"] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 2]
        # print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
        total_results["test_accuracy"] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 3]
        # print(f'Correct Labeled Train Accuracy: {r.mean():.2f} ± {r.std():.2f}')
        total_results["correct_labeled_train_accuracy"] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 4]
        # print(f'Incorrect Labeled Train Accuracy: {r.mean():.2f} ± {r.std():.2f}')
        total_results['incorrect_labeled_train_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 5]
        # print(f'Incorrect Labeled Mislead Train Accuracy: {r.mean():.2f} ± {r.std():.2f}')
        total_results['incorrect_labeled_mislead_train_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 6]
        # print(f'Unlabeled Correct Supervised Accuracy: {r.mean():.2f} ± {r.std():.2f}')
        total_results['unlabeled_correct_supervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 7]
        # print(f'Unlabeled Unsupervised Accuracy: {r.mean():.2f} ± {r.std():.2f}')
        total_results['unlabeled_unsupervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 8]
        # print(f'Unlabeled Incorrect Supervised Accuracy: {r.mean():.2f} ± {r.std():.2f}')
        total_results['unlabeled_incorrect_supervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 9] * 0.01
        # print(f'Time: {r.mean():.2f} ± {r.std():.2f}')
        total_results['total_time'] = {"mean": r.mean(), "std": r.std()}

        return total_results


class ResultLogger(object):
    def __init__(self, method_list, data_list, noise_list, runs):
        self.file_name = str(time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_path = './log/' + self.file_name + '.txt'
        self.tex_path = './log/' + self.file_name + '.tex'
        self.excel_path = './log/' + self.file_name + '.xlsx'
        self.runs = runs
        columns = method_list
        tex_index = [data_list,
                     [f'$ {int(noise_list[i][0] * 100):2d} \\% $ {noise_list[i][1]}' for i in range(len(noise_list))]]
        tex_index = pd.MultiIndex.from_product(tex_index, names=["Dataset", "Noise type"])
        excel_index = [data_list,
                       [f'{int(noise_list[i][0] * 100):2d} % {noise_list[i][1]}' for i in range(len(noise_list))]]
        excel_index = pd.MultiIndex.from_product(excel_index, names=["Dataset", "Noise type"])
        self.tex_result_tabel = pd.DataFrame(index=tex_index, columns=columns)
        self.excel_result_tabel_main = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_ave = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_std = pd.DataFrame(index=excel_index, columns=columns)

    def dump_record(self, method_name, data_name, noise_type, noise_rate, test_acc, test_acc_std):
        # print results in terminal
        message = f'| data: {data_name:12s} | method: {method_name:12s}' + f' | noise type: {noise_type:12s} | noise rate: {noise_rate:03.2f} | test acc: {test_acc:03.2f} ± {test_acc_std:03.2f} |'
        print(message)
        # dump record
        with open(self.log_path, 'a') as f:
            # record results in plain text
            message = f'| data: {data_name:12s} | method: {method_name:12s}' + f' | noise type: {noise_type:12s} | noise rate: {noise_rate:03.2f} | test acc: {test_acc:03.2f} ± {test_acc_std:03.2f} |\n'
            f.write(message)
        with open(self.tex_path, 'w') as f:
            # record results in tex
            self.tex_result_tabel.loc[data_name, f'$ {int(noise_rate * 100):2d} \\% $ {noise_type}'][
                method_name] = f'$ {test_acc:03.2f} \\pm {test_acc_std:03.2f} $'
            message = self.tex_result_tabel.to_latex(na_rep='0', bold_rows=True, caption=f'RESULTS FOR {self.runs:d} RUNS')
            f.write(message)
        with pd.ExcelWriter(self.excel_path, engine='xlsxwriter') as writer:
            self.excel_result_tabel_main.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{test_acc:03.2f} ± {test_acc_std:03.2f}'
            self.excel_result_tabel_main.to_excel(excel_writer=writer, sheet_name='main', na_rep='0')

            self.excel_result_tabel_ave.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{test_acc:03.2f}'
            self.excel_result_tabel_ave.to_excel(excel_writer=writer, sheet_name='ave', na_rep='0')

            self.excel_result_tabel_std.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{test_acc_std:03.2f}'
            self.excel_result_tabel_std.to_excel(excel_writer=writer, sheet_name='std', na_rep='0')


class SingleExpRecorder:
    """
    SingleExpRecorder Class.

    This records the performances of epochs in a single run. It determines whether the training has improved based
    on the provided `criterion` and determines whether the earlystop `patience` has been achieved.

    Parameters
    ----------
    patience : int
        The maximum epochs to keep training since last improvement.
    criterion : str
        The criterion to determine whether the training has improvement.
        - ``None``: Improvement will be considered achieved in any case.
        - ``loss``: Improvement will be considered achieved when loss decreases.
        - ``metric``: Improvement will be considered achieved when metric increases.
        - ``either``: Improvement will be considered achieved if either loss decreases or metric increases.
        - ``both``: Improvement will be considered achieved if both loss decreases and metric increases.
    """
    def __init__(self, patience=100, criterion=None):
        self.patience = patience
        self.criterion = criterion
        self.best_loss = 1e8
        self.best_metric = -1
        self.wait = 0

    def add(self, loss_val, metric_val):
        '''
        Function to add the loss and metric of a new epoch.

        Parameters
        ----------
        loss_val : float
        metric_val : float

        Returns
        -------
        flag : bool
            Whether improvement has been achieved in the epoch.
        flag_earlystop: bool
            Whether training needs earlystopping.
        '''
        flag = False
        if self.criterion is None:
            flag = True
        elif self.criterion == 'loss':
            flag = loss_val < self.best_loss
        elif self.criterion == 'metric':
            flag = metric_val > self.best_metric
        elif self.criterion == 'either':
            flag = loss_val < self.best_loss or metric_val > self.best_metric
        elif self.criterion == 'both':
            flag = loss_val < self.best_loss and metric_val > self.best_metric
        else:
            raise NotImplementedError

        if flag:
            self.best_metric = metric_val
            self.best_loss = loss_val
            self.wait = 0
        else:
            self.wait += 1

        flag_earlystop = self.patience and self.wait >= self.patience

        return flag, flag_earlystop
