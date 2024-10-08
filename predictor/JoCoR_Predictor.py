from predictor.Base_Predictor import Predictor
from predictor.module.JoCoR import loss_jocor
from predictor.module.GNNs import GCN
from copy import deepcopy
import time
import torch
import numpy as np
import nni


class jocor_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model1 = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                          out_channels=conf.model['n_classes'],
                          n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                          norm_info=conf.model['norm_info'],
                          act=conf.model['act'], input_layer=conf.model['input_layer'],
                          output_layer=conf.model['output_layer']).to(self.device)
        self.model2 = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                          out_channels=conf.model['n_classes'],
                          n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                          norm_info=conf.model['norm_info'],
                          act=conf.model['act'], input_layer=conf.model['input_layer'],
                          output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.loss_fn = loss_jocor
        self.forget_rate = conf.model['forget_rate']
        self.rate_schedule = np.ones(conf.training['n_epochs']) * conf.model['forget_rate']
        self.rate_schedule[:conf.model['num_gradual']] = np.linspace(0, self.forget_rate ** conf.model['exponent'], conf.model['num_gradual'])
        self.co_lambda = conf.model['co_lambda']

    def get_prediction(self, features, adj, label=None, mask=None, epoch=None):
        if epoch is None:
            epoch = self.conf.training['n_epochs'] - 1
        output1 = self.model1(features, adj)
        output2 = self.model2(features, adj)
        logits1 = output1[mask]
        logits2 = output2[mask]

        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = self.loss_fn(logits1, logits2, label[mask], self.rate_schedule[epoch], self.co_lambda)
            acc = self.metric(label[mask].cpu().numpy(), output1[mask].detach().cpu().numpy())
        return output1, loss, acc

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model1.train()
            self.model2.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, self.train_mask, epoch)
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
                self.weights = deepcopy(self.model1.state_dict())
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
        return self.result

    def evaluate(self, label, mask):
        self.model1.eval()
        self.model2.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            _, loss, acc = self.get_prediction(features, adj, label, mask)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model1.load_state_dict(self.weights)
        label = self.clean_label
        return self.evaluate(label, mask)





