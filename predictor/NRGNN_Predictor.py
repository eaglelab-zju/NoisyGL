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

    def get_prediction(self, features, adj, label=None, mask=None, reformed_adj_model=None):
        if reformed_adj_model is None:
            edge_index = adj.indices()
            # prediction of accurate pseudo label miner
            representations, rec_loss = self.estimator(edge_index, features)
            predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index, representations)
            pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
            predictor_weights_1 = torch.cat([torch.ones([edge_index.shape[1]], device=self.device), predictor_weights],
                                            dim=0).detach()
            reformed_adj_pred = torch.sparse_coo_tensor(pred_edge_index, predictor_weights_1, [self.n_nodes, self.n_nodes])
            log_pred = self.predictor(features, reformed_adj_pred)
            if self.best_pred == None:
                pred = F.softmax(log_pred, dim=1).detach()
                self.best_pred = pred
                self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
            else:
                pred = self.best_pred
            estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index, representations)
            estimated_weights_1 = torch.cat([predictor_weights_1, estimated_weights], dim=0).detach()
            model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)
            reformed_adj_model = torch.sparse_coo_tensor(model_edge_index, estimated_weights_1,
                                                     [self.n_nodes, self.n_nodes])
            output = self.model(features, reformed_adj_model)
            pred_model = F.softmax(output, dim=1)
            eps = 1e-8
            pred_model = pred_model.clamp(eps, 1 - eps)
            # loss from pseudo labels
            loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()
            # loss of accurate pseudo label miner
            loss_pred = self.loss_fn(log_pred[mask], label[mask])
            # loss of GCN classifier
            loss_gcn = self.loss_fn(output[mask], label[mask])
            loss = loss_gcn + loss_pred + self.conf.model['alpha'] * rec_loss + self.conf.model['beta'] * loss_add
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            return output, loss, acc, estimated_weights_1, model_edge_index, predictor_weights_1, pred
        else:
            # loss of GCN classifier
            output = self.model(features, reformed_adj_model)
            loss_gcn = self.loss_fn(output[mask], label[mask])
            loss = loss_gcn
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            return output, loss, acc


    def train(self):
        for epoch in range(self.conf.training['n_epochs']):

            improve = ''
            t0 = time.time()
            self.model.train()
            self.predictor.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            _, loss_train, acc_train, estimated_weights, model_edge_index, predictor_weights, pred \
                = self.get_prediction(features, adj, self.noisy_label, self.train_mask)
            loss_train.backward()
            self.optim.step()
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask, model_edge_index, estimated_weights)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train

                self.best_val_acc = acc_val
                self.best_graph = estimated_weights.detach()
                self.best_model_index = model_edge_index
                self.weights = deepcopy(self.model.state_dict())

                # self.best_acc_pred_val = acc_pred_val
                self.best_pred_graph = predictor_weights.detach()
                self.best_pred = pred.detach()
                self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
                self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)
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

    def evaluate(self, label, mask, model_edge_index=None, estimated_weights=None):
        features = self.feats
        adj = self.adj
        if model_edge_index is None:
            estimated_weights = self.best_graph
            model_edge_index = self.best_model_index
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            reformed_adj_model = torch.sparse_coo_tensor(model_edge_index, estimated_weights,
                                                         [self.n_nodes, self.n_nodes])
            _, loss, acc = self.get_prediction(features, adj, label, mask, reformed_adj_model)
        return loss, acc

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
