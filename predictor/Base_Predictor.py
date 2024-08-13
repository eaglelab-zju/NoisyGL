import time
import torch
from utils.functional import accuracy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.recorder import Recorder
from copy import deepcopy
import nni


class Predictor:
    def __init__(self, conf, data, device='cuda:0'):
        super(Predictor, self).__init__()
        self.conf = conf
        self.device = torch.device(device)
        self.general_init(conf, data)
        self.method_init(conf, data)

    def general_init(self, conf, data):
        '''
        This conducts necessary operations for an experiment, including the setting specified split,
        variables to record statistics.
        '''
        self.loss_fn = F.binary_cross_entropy_with_logits if data.n_classes == 1 else F.cross_entropy
        self.metric = roc_auc_score if data.n_classes == 1 else accuracy
        self.edge_index = data.adj.indices()
        self.adj = data.adj if self.conf.dataset['sparse'] else data.adj.to_dense()
        self.recoder = Recorder(self.conf.training['patience'], self.conf.training['criterion'])
        self.feats = data.feats
        self.n_nodes = data.n_nodes
        self.n_classes = data.n_classes
        self.clean_label = data.labels
        self.noisy_label = data.noisy_label
        self.train_mask = data.train_masks
        self.val_mask = data.val_masks
        self.test_mask = data.test_masks
        self.result = {'train': -1, 'valid': -1, 'test': -1}
        self.weights = None
        self.start_time = time.time()
        self.total_time = -1

    def method_init(self, conf, data):
        '''
        This sets module and other members, which is overwritten for each method.
        '''
        return None

    def train(self):
        '''
        This is the common learning procedure, which is overwritten for special learning procedure.

        Parameters
        ----------
        None

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        '''

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            # forward and backward
            output = self.model(features, adj)

            loss_train = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())

            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.noisy_label[self.val_mask], self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                loss_test, acc_test = self.test(self.test_mask)
                nni.report_intermediate_result(acc_test)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask):
        '''
        This is the common evaluation procedure, which is overwritten for special evaluation procedure.

        Parameters
        ----------
        label : torch.tensor
        mask: torch.tensor

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            output = self.model(features, adj)
        logits = output[mask]
        loss = self.loss_fn(logits, label)
        return loss, self.metric(label.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self, mask):
        '''
        This is the common test procedure, which is overwritten for special test procedure.

        Returns
        -------
        loss : float
            Test loss.
        metric : float
            Test metric.
        '''
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        label = self.clean_label[mask]
        return self.evaluate(label, mask)







