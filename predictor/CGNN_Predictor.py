from predictor.Base_Predictor import Predictor
from predictor.module.CGNN import *
import torch
import time
from copy import deepcopy
from torch_geometric.utils import dropout_adj


class cgnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = CGNN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                          out_channels=conf.model['n_classes'], dropout=conf.model['dropout']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.temperature = 0.5

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            features, adj = self.feats, self.edge_index
            if epoch >= self.conf.training['warmup_epochs']:
                cleaned_labels = clean_noisy_labels(self.model, features, self.adj, self.noisy_label, self.train_mask,
                                                    self.n_nodes, threshold=0.8)
                self.noisy_label[self.train_mask] = cleaned_labels[self.train_mask]
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            output = self.model(features, adj)
            edge_index, _ = dropout_adj(adj, p=0.5, force_undirected=True, training=True)
            out_edge_dropout = self.model(features, edge_index)
            x_node_dropout = custom_dropout_features(features, p=0.5, training=True)
            out_node_dropout = self.model(x_node_dropout, adj)

            loss_cl = contrastive_loss(out_edge_dropout, out_node_dropout, temperature=0.5)
            loss_sup = supervised_loss(out_edge_dropout[self.train_mask], self.noisy_label[self.train_mask])
            loss_train = loss_cl + loss_sup
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
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result
