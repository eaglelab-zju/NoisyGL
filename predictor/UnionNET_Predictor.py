from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
from predictor.module.UnionNET import *
import torch
import time
from copy import deepcopy
import nni


class unionnet_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        # UnionNET
        self.k = conf.model['k']
        self.alpha = conf.model['alpha']
        self.beta = conf.model['beta']
        self.kldiv = torch.nn.KLDivLoss()

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj

            # forward and backward
            output = self.model(features, adj)
            support_set, labels = construct_support_set(features, self.train_mask, self.noisy_label, self.edge_index, self.k)
            support_set = support_set[self.train_mask]
            labels = labels[self.train_mask].long()
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=self.n_classes).to(self.device)
            class_probabilities = label_aggregation(support_set, one_hot_labels, features[self.train_mask])
            class_prob = class_probabilities[0][self.noisy_label[self.train_mask]]

            loss_reweighting = torch.sum(class_prob * F.cross_entropy(output[self.train_mask], self.noisy_label[self.train_mask]))
            loss_correction = label_correction_loss(support_set, self.noisy_label[self.train_mask], features[self.train_mask], class_prob)
            # Add kl-divergence loss results in a worse performance, we set beta = 0
            # loss_kl = F.kl_div(output[self.train_mask], F.one_hot(self.noisy_label[self.train_mask]).to(torch.float))
            # loss_train = self.alpha * loss_reweighting + (1 - self.alpha) * loss_correction + self.beta * loss_kl
            loss_train = self.alpha * loss_reweighting + (1 - self.alpha) * loss_correction
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())

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
        return self.result