import numpy as np
import torch
import torch.nn.functional as F
from predictor.module.R2LP import R2LP, new_clean
from predictor.Base_Predictor import Predictor
import time
from copy import deepcopy
import nni


class r2lp_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = R2LP(nnodes=self.n_nodes, in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                          dropout=conf.model['dropout'],alpha=conf.model['alpha'], alpha1=conf.model['alpha1'], alpha2=conf.model['alpha2'], alpha3=conf.model['alpha3'],
                          beta=conf.model['beta'], gamma=conf.model['gamma'], delta=conf.model['delta'], norm_func_id=conf.model['norm_func_id'],
                          norm_layers=conf.model['norm_layers'], orders=conf.model['orders'], orders_func_id=conf.model['orders_func_id'],
                          device=self.device).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.loss_fn = torch.nn.NLLLoss()
        self.pre_unknown = conf.training['pre_unknown']
        self.pre_select = conf.training['pre_select']

        idx_all = torch.arange(self.n_nodes).to(self.device)
        self.idx_clean = idx_all[:0]
        self.idx_unknown = idx_all[0:int(self.pre_unknown * self.n_nodes)]
        self.y_clean = torch.zeros((self.n_nodes, self.n_classes)).float().to(self.device)
        self.y_unknown = torch.zeros((self.n_nodes, self.n_classes)).to(self.device)
        self.y_unknown[self.idx_unknown] = F.one_hot(self.noisy_label[self.idx_unknown],
                                                     self.n_classes).float().squeeze(1)  # the matrix of noisy labels

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, adj, self.y_clean, self.y_unknown, if_lp=False)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        features, adj = self.feats, self.adj
        new_labels = self.noisy_label.clone()
        for epoch_lp in range(self.conf.training['n_epochs_lp']):
            for epoch in range(self.conf.training['n_epochs']):
                improve = ''
                t0 = time.time()
                self.model.train()
                self.optim.zero_grad()
                # forward and backward
                output, loss_train, acc_train = self.get_prediction(features, adj, new_labels, self.idx_clean)
                loss_train.backward()
                self.optim.step()

                # Evaluate
                loss_val, acc_val = self.evaluate(new_labels, self.val_mask)
                # loss_val, acc_val = self.evaluate(self.clean_label, self.test_mask)
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
                    nni.report_intermediate_result(acc_val)
                    print(
                        "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                            epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

            # label propagation
            self.model.load_state_dict(self.weights)
            self.model.eval()
            output, predict_z, y_predict, h_l, fl = self.model(features, adj, self.y_clean, self.y_unknown, if_lp=True)
            F_t = fl
            new_idx_unknown, new_idx_clean, labels_lp = new_clean(F_t, self.pre_select, self.idx_clean, self.idx_unknown,
                                                                  new_labels, self.y_clean)

            self.idx_unknown = new_idx_unknown
            self.idx_clean = new_idx_clean
            if self.conf.training['debug']:
                print('num_celan:', self.idx_clean.shape[0], 'num_unknown:', self.idx_unknown.shape[0])
            new_labels = labels_lp

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result