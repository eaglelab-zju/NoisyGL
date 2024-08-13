from predictor.Base_Predictor import Predictor
from predictor.module.Smodel import NoiseAda
from predictor.module.GNNs import GCN
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.tools import heatmap
import nni


class smodel_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'], norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'], output_layer=conf.model['output_layer']).to(self.device)
        self.noiseAda = NoiseAda(n_class=conf.model['n_classes'], noise_rate_init=conf.model['noise_rate_init']).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.noiseAda.parameters()),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.noiseAda.train()
            self.optim.zero_grad()
            features = self.feats
            adj = self.adj

            # forward and backward
            output = self.model(features, adj)
            pred = F.softmax(output, dim=1)
            eps = 1e-8
            score = self.noiseAda(pred).clamp(eps, 1 - eps)

            loss_train = self.loss_fn(torch.log(score[self.train_mask]), self.noisy_label[self.train_mask])
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(), output[self.train_mask].detach().cpu().numpy())

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

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test(self.test_mask)
        if self.conf.training['debug']:
            heatmap(self.noiseAda.B.detach().cpu().numpy(), n_classes=self.n_classes, title="S-model")
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result
