from predictor.Base_Predictor import Predictor
from predictor.module.NRGNN import EstimateAdj
from predictor.module.GNNs import GCN
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
import torch_geometric.utils as utils
import nni


class nrgnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.predictor = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                             out_channels=conf.model['n_classes'],
                             n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                             norm_info=conf.model['norm_info'],
                             act=conf.model['act'], input_layer=conf.model['input_layer'],
                             output_layer=conf.model['output_layer']).to(self.device)

        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)

        self.estimator = EstimateAdj(conf).to(self.device)

        self.optim = torch.optim.Adam(
            list(self.model.parameters()) + list(self.estimator.parameters()) + list(self.predictor.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])

        # NRGNN
        self.best_pred = None
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        edge_index = self.adj.indices()
        features = self.feats
        self.edge_index = edge_index.to(self.device)
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(self.train_mask))).to(self.device)

        self.pred_edge_index = self.get_train_edge(edge_index, features, self.conf.model['n_p'], self.train_mask)

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):

            improve = ''
            t0 = time.time()
            self.model.train()
            self.predictor.train()
            self.optim.zero_grad()

            # obtain representations and rec loss of the estimator
            features, adj = self.feats, self.adj
            edge_index = adj.indices()

            # prediction of accurate pseudo label miner
            representations, rec_loss = self.estimator(edge_index, features)
            predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index, representations)
            pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
            predictor_weights_1 = torch.cat([torch.ones([edge_index.shape[1]], device=self.device), predictor_weights],
                                            dim=0).detach()

            reformed_adj_1 = torch.sparse_coo_tensor(pred_edge_index, predictor_weights_1, [self.n_nodes, self.n_nodes])

            log_pred = self.predictor(features, reformed_adj_1)

            if self.best_pred == None:

                pred = F.softmax(log_pred, dim=1).detach()
                self.best_pred = pred
                self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
            else:
                pred = self.best_pred

            estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index, representations)
            estimated_weights_1 = torch.cat([predictor_weights_1, estimated_weights], dim=0).detach()
            model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)
            reformed_adj_2 = torch.sparse_coo_tensor(model_edge_index, estimated_weights_1,
                                                     [self.n_nodes, self.n_nodes])

            output = self.model(features, reformed_adj_2)
            pred_model = F.softmax(output, dim=1)

            eps = 1e-8
            pred_model = pred_model.clamp(eps, 1 - eps)

            # loss from pseudo labels
            loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()

            # loss of accurate pseudo label miner
            loss_pred = self.loss_fn(log_pred[self.train_mask], self.noisy_label[self.train_mask])

            # loss of GCN classifier
            loss_gcn = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])

            total_loss = loss_gcn + loss_pred + self.conf.model['alpha'] * rec_loss + self.conf.model['beta'] * loss_add

            total_loss.backward()

            self.optim.step()

            # forward and backward
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())

            # Evaluate validation set performance separately
            # acc_pred_val, acc_val, loss_val = self.evaluate(self.noisy_label[self.val_mask], self.val_mask, pred_edge_index, predictor_weights_1, model_edge_index, estimated_weights_1)
            loss_val, acc_val = self.evaluate(self.noisy_label[self.val_mask], self.val_mask, pred_edge_index,
                                              predictor_weights_1, model_edge_index, estimated_weights_1)

            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train

                self.best_val_acc = acc_val
                self.best_graph = estimated_weights_1.detach()
                self.best_model_index = model_edge_index
                self.weights = deepcopy(self.model.state_dict())

                # self.best_acc_pred_val = acc_pred_val
                self.best_pred_graph = predictor_weights_1.detach()
                self.best_pred = pred.detach()
                self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
                self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, total_loss.item(), acc_train, loss_val, acc_val, improve))

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask, pred_edge_index=None, predictor_weights=None, model_edge_index=None, estimated_weights=None):
        if pred_edge_index is None:
            predictor_weights = self.best_pred_graph
            pred_edge_index = torch.cat([self.edge_index, self.pred_edge_index], dim=1)
            estimated_weights = self.best_graph
            model_edge_index = self.best_model_index
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            features = self.feats
            reformed_adj_1 = torch.sparse_coo_tensor(pred_edge_index, predictor_weights, [self.n_nodes, self.n_nodes])
            reformed_adj_2 = torch.sparse_coo_tensor(model_edge_index, estimated_weights, [self.n_nodes, self.n_nodes])
            with torch.no_grad():
                pred = F.softmax(self.predictor(features, reformed_adj_1), dim=1)
                output = self.model(features, reformed_adj_2)
            logits = output[mask]
            loss_val = self.loss_fn(logits, label)

            acc_pred_val = self.metric(label.cpu().numpy(), pred[mask].detach().cpu().numpy())
            acc_val = self.metric(label.cpu().numpy(), output[mask].detach().cpu().numpy())
        return loss_val, acc_val

    def test(self, mask):
        features = self.feats
        labels = self.clean_label
        # edge_index = self.adj.indices()
        edge_index = self.edge_index
        idx_test = mask
        self.predictor.eval()
        if self.predictor_model_weigths is not None:
            self.predictor.load_state_dict(self.predictor_model_weigths)
        with torch.no_grad():
            estimated_weights = self.best_pred_graph
            pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
            reformed_adj_1 = torch.sparse_coo_tensor(pred_edge_index, estimated_weights, [self.n_nodes, self.n_nodes])

            output = self.predictor(features, reformed_adj_1)
            loss_test = self.loss_fn(output[idx_test], labels[idx_test])
            acc_test = self.metric(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
            if self.conf.training["debug"]:
                print("\tPredictor results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

            self.model.eval()
            self.model.load_state_dict(self.weights)
            estimated_weights = self.best_graph
            model_edge_index = self.best_model_index
            reformed_adj_2 = torch.sparse_coo_tensor(model_edge_index, estimated_weights, [self.n_nodes, self.n_nodes])

            output = self.model(features, reformed_adj_2)
            loss_test = self.loss_fn(output[idx_test], labels[idx_test])
            acc_test = self.metric(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
            if self.conf.training["debug"]:
                print("\tGCN classifier results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        return loss_test, acc_test

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        '''
        obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
        n_p is the top n_p labeled nodes similar with unlabeled nodes
        '''

        if n_p == 0:
            return None

        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1, edge_index[0] == i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i], features[idx_train].T),
                                features[i].norm() * features[idx_train].norm(dim=1))
                _, rank = sim.topk(n_p)
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices)
                else:
                    indices = set()
                indices = indices - set(edge_index[1, edge_index[0] == i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges, len(features)).to(self.device)

        return poten_edges

    def get_model_edge(self, pred):

        idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.conf.model['p_u'])]

        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel), 1).T.flatten()
        mask = (row != col)
        unlabel_edge_index = torch.stack([row[mask], col[mask]], dim=0)

        return unlabel_edge_index, idx_add
