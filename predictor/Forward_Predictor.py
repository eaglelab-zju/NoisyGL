from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
from predictor.module.DGNN import estimate_C, forward_correction_xentropy
import time
import torch
from copy import deepcopy
import numpy as np
from utils.tools import heatmap
import nni


class forward_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        '''
        This sets module and other members, which is overwritten for each method.
        '''
        self.pre_model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.pre_optim = torch.optim.Adam(self.pre_model.parameters(), lr=conf.training['lr'],
                                          weight_decay=conf.training['weight_decay'])

        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=conf.training['lr'],
                                      weight_decay=conf.training['weight_decay'])
        self.C = np.zeros((self.n_classes, self.n_classes), dtype=float)


    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = forward_correction_xentropy(output[mask], self.noisy_label[mask],
                                               self.C, self.device, self.n_classes)
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        best_pre_acc = 0
        for pre_epoch in range(self.conf.training['n_pre_epochs']):
            self.pre_model.train()
            self.pre_optim.zero_grad()
            features, adj = self.feats, self.adj

            # forward and backward
            output = self.pre_model(features, adj)

            loss_train = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            if acc_train > best_pre_acc:
                best_pre_acc = acc_train
                pre_weight = self.pre_model.state_dict()

            loss_train.backward()
            self.pre_optim.step()
            if self.conf.training['debug']:
                print("pre_Epoch {:05d} | pre_Loss(train) {:.4f} | pre_Acc(train) {:.4f} | ".format(
                        pre_epoch + 1, loss_train.item(), acc_train))
        self.pre_model.load_state_dict(pre_weight)
        self.C = estimate_C(model=self.pre_model, x=features, adj=adj, n_classes=self.n_classes)

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj

            # forward and backward
            # output = self.model(features, adj)

            # loss_train = forward_correction_xentropy(output[self.train_mask],
            #                         self.noisy_label[self.train_mask], self.C, self.device, self.n_classes)
            # loss_train = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
            # acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
            #                         output[self.train_mask].detach().cpu().numpy())
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, self.train_mask)
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
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

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
            # heatmap(self.C, n_classes=self.n_classes, title="Backward")
        return self.result

    # def evaluate(self, label, mask):
    #     '''
    #     This is the common evaluation procedure, which is overwritten for special evaluation procedure.
    #
    #     Parameters
    #     ----------
    #     label : torch.tensor
    #     mask: torch.tensor
    #
    #     Returns
    #     -------
    #     loss : float
    #         Evaluation loss.
    #     metric : float
    #         Evaluation metric.
    #     '''
    #     self.model.eval()
    #     features, adj = self.feats, self.adj
    #     with torch.no_grad():
    #         output = self.model(features, adj)
    #     logits = output[mask]
    #     loss = forward_correction_xentropy(logits, label, self.C, self.device, self.n_classes)
    #     return loss, self.metric(label.cpu().numpy(), logits.detach().cpu().numpy())
    #
    # def test(self, mask):
    #     '''
    #     This is the common test procedure, which is overwritten for special test procedure.
    #
    #     Returns
    #     -------
    #     loss : float
    #         Test loss.
    #     metric : float
    #         Test metric.
    #     '''
    #     self.model.load_state_dict(self.weights)
    #     label = self.clean_label[mask]
    #     return self.evaluate(label, mask)
