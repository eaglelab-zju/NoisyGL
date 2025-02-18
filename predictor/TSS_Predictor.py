import numpy as np

from predictor.Base_Predictor import Predictor
from predictor.module.TSS import *
from predictor.module.GNNs import GCN
import torch
import torch.nn.functional as F
import time
from copy import deepcopy
import nni



class tss_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):

        Pi = Personalized_PageRank(conf.model["pagerank-prob"], self.adj)
        data.Pi = Pi.to(self.device)
        node_difficulty = difficulty_measurer(data, self.train_mask).to(self.device)
        # handling outliers
        node_difficulty[node_difficulty == 0] = float("inf")
        # sort dataset by CBC
        _, indices = torch.sort(node_difficulty[self.train_mask])
        indices = indices.cpu().numpy()
        self.sorted_trainset = self.train_mask[indices]
        pre_model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                        out_channels=conf.model['n_classes'],
                        n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                        norm_info=conf.model['norm_info'],
                        act=conf.model['act'], input_layer=conf.model['input_layer'],
                        output_layer=conf.model['output_layer']).to(self.device)
        pre_optimizer = torch.optim.Adam(pre_model.parameters(), lr=conf.training['lr'],
                                         weight_decay=conf.training['weight_decay'])
        for _ in range(conf.training['pretrain_epoch']):
            pre_model.train()
            pre_optimizer.zero_grad()
            output = pre_model(self.feats, self.adj)
            loss = F.cross_entropy(output[self.train_mask], self.noisy_label[self.train_mask])
            loss.backward()
            pre_optimizer.step()
        self.pretrain_pred = torch.argmax(pre_model(self.feats, self.adj), dim=1).cpu().numpy()
        self.lam = conf.model['lam']
        self.T = conf.model['T']
        self.scheduler = conf.model['scheduler']



        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])



    def train(self):
        sorted_trainset = self.sorted_trainset
        pretrain_pred = self.pretrain_pred

        train_subset = sorted_trainset[:int(self.lam * sorted_trainset.shape[0])]
        mask = pretrain_pred[train_subset] == self.noisy_label[train_subset].cpu().numpy()
        clean_idx = train_subset[pretrain_pred[train_subset] == self.noisy_label[train_subset].cpu().numpy()]
        unlabeled_idx = train_subset[pretrain_pred[train_subset] != self.noisy_label[train_subset].cpu().numpy()]
        train_subset = clean_idx
        add_set = np.empty(0, dtype=int)
        later_size = self.lam

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            train_subset = np.concatenate((train_subset, add_set))
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            # forward and backward

            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, train_subset)
            loss_train.backward()
            self.optim.step()

            if epoch < self.conf.model['T']:
                size = training_scheduler(self.lam, epoch+1, self.T, scheduler=self.scheduler)
                add_set = sorted_trainset[int(later_size * sorted_trainset.shape[0]):int(size * sorted_trainset.shape[0])]
                clean_idx = add_set[pretrain_pred[add_set] == self.noisy_label[add_set].cpu().numpy()]
                unlabeled_idx = add_set[pretrain_pred[add_set] != self.noisy_label[add_set].cpu().numpy()]
                add_set =  clean_idx
                later_size = size
            else:
                add_set = np.empty(0, dtype=int)

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