import torch
import time
import pandas as pd


class MultiExpRecorder(object):
    """
    MultiExpRecorder Class.
    This records the performances of multiple runs.

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
        This function computes the mean and standard deviation of the results across all runs.

        Parameters
        ----------
            None

        Returns
        -------
        total_results : dict
            The statistics of a given run or all runs.

        '''
        best_result = 100 * torch.tensor(self.results)
        total_results = {}
        r = best_result[:, 0]
        total_results["train_accuracy"] = {"acc": r.mean(), "std": r.std}

        r = best_result[:, 1]
        total_results["valid_accuracy"] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 2]
        total_results["test_accuracy"] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 3]
        total_results["correct_labeled_train_accuracy"] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 4]
        total_results['incorrect_labeled_train_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 5]
        total_results['incorrect_labeled_mislead_train_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 6]
        total_results['unlabeled_correct_supervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 7]
        total_results['unlabeled_unsupervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 8]
        total_results['unlabeled_incorrect_supervised_accuracy'] = {"acc": r.mean(), "std": r.std()}

        r = best_result[:, 9] * 0.01

        total_results['total_time'] = {"mean": r.mean(), "std": r.std()}

        return total_results


class ResultLogger(object):
    """
    ResultLogger Class.
    This records the performances of multiple runs in a structured way, including plain text, LaTeX and Excel files.

    Parameters
    ----------
    method_list : list
        List of method names.
    data_list : list
        List of dataset names.
    noise_list : list
        List of noise types and rates, where each element is a tuple (rate, type).
    runs : int
        Total experimental runs.
    """
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
        self.excel_result_tabel_test_acc_main = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_test_acc_ave = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_test_acc_std = pd.DataFrame(index=excel_index, columns=columns)

        self.excel_result_tabel_aclt = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_ailt = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_aucs = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_auis = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_auu = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_ailmt = pd.DataFrame(index=excel_index, columns=columns)
        self.excel_result_tabel_time = pd.DataFrame(index=excel_index, columns=columns)

    def dump_record(self, method_name, data_name, noise_type, noise_rate, total_results):
        '''
        Function to dump the record of a single run.
        This function records the results of a single run in plain text, LaTeX and Excel files.

        Parameters
        ----------
        method_name: str
            Name of the method.
        data_name: str
            Name of the dataset.
        noise_type: str
            Type of the noise, e.g., 'uniform', 'pair', etc.
        noise_rate: float
            Rate of the noise, e.g., 0.3 for 30% noise.
        total_results: dict
            A dictionary containing the results of the experiment, including test accuracy, time, etc.

        Returns
        -------
        None

        '''
        test_acc_ave, test_acc_std = total_results['test_accuracy']['acc'], total_results['test_accuracy']['std']
        aclt_ave, aclt_std = (total_results["correct_labeled_train_accuracy"]['acc'],
                              total_results["correct_labeled_train_accuracy"]['std'])
        ailt_ave, ailt_std = (total_results["incorrect_labeled_train_accuracy"]['acc'],
                              total_results["incorrect_labeled_train_accuracy"]['std'])
        aucs_ave, aucs_std = (total_results['unlabeled_correct_supervised_accuracy']['acc'],
                              total_results['unlabeled_correct_supervised_accuracy']['std'])
        auu_ave, auu_std = (total_results['unlabeled_unsupervised_accuracy']['acc'],
                              total_results['unlabeled_unsupervised_accuracy']['std'])
        auis_ave, auis_std = (total_results['unlabeled_incorrect_supervised_accuracy']['acc'],
                              total_results['unlabeled_incorrect_supervised_accuracy']['std'])
        ailmt_ave, ailmt_std = (total_results['incorrect_labeled_mislead_train_accuracy']['acc'],
                              total_results['incorrect_labeled_mislead_train_accuracy']['std'])
        time_ave, time_std = (total_results['total_time']['mean'],
                              total_results['total_time']['std'])

        # print results in terminal
        message = f'| data: {data_name:12s} | method: {method_name:12s}' + f' | noise type: {noise_type:12s} | noise rate: {noise_rate:03.2f} | test acc: {test_acc_ave:03.2f} ± {test_acc_std:03.2f} |'
        print(message)
        # dump record
        with open(self.log_path, 'a') as f:
            # record results in plain text
            # dump test acc main results
            message = f'| data: {data_name:12s} | method: {method_name:12s}' + f' | noise type: {noise_type:12s} | noise rate: {noise_rate:03.2f} | test acc: {test_acc_ave:03.2f} ± {test_acc_std:03.2f} |\n'
            f.write(message)
        with open(self.tex_path, 'w') as f:
            # record results in tex file
            # dump test acc main results
            self.tex_result_tabel.loc[data_name, f'$ {int(noise_rate * 100):2d} \\% $ {noise_type}'][
                method_name] = f'$ {test_acc_ave:03.2f} \\pm {test_acc_std:03.2f} $'
            message = self.tex_result_tabel.to_latex(na_rep='0', bold_rows=True, caption=f'RESULTS FOR {self.runs:d} RUNS')
            f.write(message)
        with pd.ExcelWriter(self.excel_path, engine='xlsxwriter') as writer:
            # record results in excel file

            # dump test acc main results
            self.excel_result_tabel_test_acc_main.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{test_acc_ave:03.2f} ± {test_acc_std:03.2f}'
            self.excel_result_tabel_test_acc_main.to_excel(excel_writer=writer, sheet_name='test_acc_main', na_rep='0')

            # dump test acc mean results
            self.excel_result_tabel_test_acc_ave.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{test_acc_ave:03.2f}'
            self.excel_result_tabel_test_acc_ave.to_excel(excel_writer=writer, sheet_name='test_acc_ave', na_rep='0')

            # dump test acc std results
            self.excel_result_tabel_test_acc_std.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{test_acc_std:03.2f}'
            self.excel_result_tabel_test_acc_std.to_excel(excel_writer=writer, sheet_name='test_acc_std', na_rep='0')

            # dump aclt results
            self.excel_result_tabel_aclt.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{aclt_ave:03.2f} ± {aclt_std:03.2f}'
            self.excel_result_tabel_aclt.to_excel(excel_writer=writer, sheet_name='aclt', na_rep='0')

            # dump ailt results
            self.excel_result_tabel_ailt.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{ailt_ave:03.2f} ± {ailt_std:03.2f}'
            self.excel_result_tabel_ailt.to_excel(excel_writer=writer, sheet_name='ailt', na_rep='0')

            # dump aucs results
            self.excel_result_tabel_aucs.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{aucs_ave:03.2f} ± {aucs_std:03.2f}'
            self.excel_result_tabel_aucs.to_excel(excel_writer=writer, sheet_name='aucs', na_rep='0')

            # dump auis results
            self.excel_result_tabel_auis.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{auis_ave:03.2f} ± {auis_std:03.2f}'
            self.excel_result_tabel_auis.to_excel(excel_writer=writer, sheet_name='auis', na_rep='0')

            # dump auu results
            self.excel_result_tabel_auu.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{auu_ave:03.2f} ± {auu_std:03.2f}'
            self.excel_result_tabel_auu.to_excel(excel_writer=writer, sheet_name='auu', na_rep='0')

            # dump ailmt results
            self.excel_result_tabel_ailmt.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{ailmt_ave:03.2f} ± {ailmt_std:03.2f}'
            self.excel_result_tabel_ailmt.to_excel(excel_writer=writer, sheet_name='ailmt', na_rep='0')

            # dump time results
            self.excel_result_tabel_time.loc[data_name, f'{int(noise_rate * 100):2d} % {noise_type}'][
                method_name] = f'{time_ave:03.2f} ± {time_std:03.2f}'
            self.excel_result_tabel_time.to_excel(excel_writer=writer, sheet_name='time', na_rep='0')



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
