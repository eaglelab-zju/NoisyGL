from predictor.module.RTGNN import *
from predictor.Base_Predictor import Predictor
import torch.nn.functional as F
from copy import deepcopy
import time


class rtgnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):

        self.predictor = Dual_GCN(n_feat=conf.model['n_feat'], n_hid=conf.model['n_hidden'],
                                  n_class=conf.model['n_classes'],
                                  n_layer=conf.model['n_layer'], dropout=conf.model['dropout'],
                                  norm_info=conf.model['norm_info'],
                                  act=conf.model['act'], input_layer=conf.model['input_layer'],
                                  output_layer=conf.model['output_layer'], n_nodes=self.n_nodes).to(self.device)
        '''
        self.predictor = Dual_GCN(nfeat=conf.model['n_feat'], nhid=conf.model['n_hidden'],
                          nclass=conf.model['n_classes'],
                          self_loop=True, dropout=conf.model['dropout']).to(self.device)
        '''
        self.estimator = EstimateAdj(conf=conf).to(self.device)
        # RTGNN
        self.best_pred = None
        self.best_graph = None
        self.criterion = LabeledDividedLoss(conf)
        self.criterion_pse = PseudoLoss()
        self.intra_reg = IntraviewReg(device=self.device)

        edge_index = self.adj.indices()
        features = self.feats
        self.edge_index = edge_index
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(self.train_mask))).to(
            self.device)

        self.pred_edge_index = self.KNN(self.edge_index, features, self.conf.model['K'], self.train_mask)
        self.optim = torch.optim.Adam(
                                list(self.estimator.parameters()) + list(self.predictor.parameters()),
                                lr=self.conf.training['lr'],
                                weight_decay=self.conf.training['weight_decay'])

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.predictor.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            edge_index = self.edge_index

            representations, rec_loss = self.estimator(features, adj)
            pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
            origin_w = torch.cat([torch.ones(edge_index.shape[1]), torch.zeros(self.pred_edge_index.shape[1])]).to(
                self.device)

            predictor_weights, _ = self.estimator.get_estimated_weigths(pred_edge_index, representations, origin_w)
            edge_remain_idx = torch.where(predictor_weights != 0)[0].detach()

            # use detach option so that the code can run on pytorch 2.0+
            predictor_weights = predictor_weights[edge_remain_idx].detach()
            pred_edge_index = pred_edge_index[:, edge_remain_idx]

            log_pred_0, log_pred_1 = self.predictor(features, pred_edge_index, predictor_weights)

            pred = F.softmax(log_pred_0, dim=1).detach()
            pred1 = F.softmax(log_pred_1, dim=1).detach()

            self.idx_add = self.get_pseudo_label(pred, pred1)

            if epoch == 0:
                loss_pred = (F.cross_entropy(log_pred_0[self.train_mask], self.noisy_label[self.train_mask]) +
                             F.cross_entropy(log_pred_1[self.train_mask], self.noisy_label[self.train_mask]))
            else:
                loss_pred = self.criterion(log_pred_0[self.train_mask], log_pred_1[self.train_mask],
                                           self.noisy_label[self.train_mask],
                                           co_lambda=self.conf.model['co_lambda'], epoch=epoch)

            if len(self.idx_add) != 0:
                loss_add = self.criterion_pse(log_pred_0, log_pred_1, self.idx_add,
                                              co_lambda=self.conf.model['co_lambda'])
            else:
                loss_add = torch.Tensor([0]).to(self.device)

            neighbor_kl_loss = self.intra_reg(log_pred_0, log_pred_1,
                                              torch.tensor(self.train_mask).to(self.device),
                                              pred_edge_index,
                                              predictor_weights)
            total_loss = (loss_pred + self.conf.model['alpha'] * rec_loss + loss_add +
                          self.conf.model['co_lambda'] * (neighbor_kl_loss))

            total_loss.backward()
            self.optim.step()

            acc_pred_train_0 = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                           log_pred_0[self.train_mask].detach().cpu().numpy())
            acc_pred_train_1 = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                           log_pred_1[self.train_mask].detach().cpu().numpy())

            acc_train = (acc_pred_train_0 + acc_pred_train_1) * 0.5

            loss_val, acc_val = self.evaluate(self.noisy_label[self.val_mask], self.val_mask,
                                              pred_edge_index, predictor_weights)

            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train

                # RTGNN
                self.best_pred_graph = predictor_weights.detach()
                self.best_edge_idx = pred_edge_index.detach()
                self.best_pred = pred.detach()
                self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, total_loss.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask, pred_edge_index=None, estimated_weights=None):
        if pred_edge_index is None:
            estimated_weights = self.best_pred_graph
            pred_edge_index = self.best_edge_idx
        self.predictor.eval()
        features = self.feats
        output_0, output_1 = self.predictor(features, pred_edge_index, estimated_weights)
        loss = (F.cross_entropy(output_0[mask], label) + F.cross_entropy(output_1[mask], label))
        acc_pred_0 = self.metric(label.cpu().numpy(), output_0[mask].detach().cpu().numpy())
        acc_pred_1 = self.metric(label.cpu().numpy(), output_1[mask].detach().cpu().numpy())
        acc_val = 0.5 * (acc_pred_0 + acc_pred_1)
        return loss, acc_val

    def test(self, mask):
        estimated_weights = self.best_pred_graph
        pred_edge_index = self.best_edge_idx
        if self.predictor_model_weigths is not None:
            self.predictor.load_state_dict(self.predictor_model_weigths)
        loss_test, acc_pred_test = self.evaluate(self.clean_label[mask], mask, pred_edge_index, estimated_weights)
        '''
        self.predictor.eval()
        self.predictor.load_state_dict(self.predictor_model_weigths)
        features = self.feats
        estimated_weights = self.best_pred_graph
        pred_edge_index = self.best_edge_idx
        output_0, output_1 = self.predictor(features, pred_edge_index, estimated_weights)
        acc_pred_test_0 = self.metric(self.clean_label[self.test_mask].cpu().numpy(),
                                      output_0[self.test_mask].detach().cpu().numpy())
        acc_pred_test_1 = self.metric(self.clean_label[self.test_mask].cpu().numpy(),
                                      output_1[self.test_mask].detach().cpu().numpy())
        acc_pred_test = 0.5 * (acc_pred_test_0 + acc_pred_test_1)
        loss_test = (F.cross_entropy(output_0[self.test_mask], self.clean_label[self.test_mask]) +
                     F.cross_entropy(output_1[self.test_mask], self.clean_label[self.test_mask]))
        '''
        return loss_test, acc_pred_test

    def KNN(self, edge_index, features, K, idx_train):
        if K == 0:
            return torch.LongTensor([])

        poten_edges = []
        if K > len(idx_train):
            for i in range(len(features)):
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in idx_train:
                sim = torch.div(torch.matmul(features[i], features[self.idx_unlabel].T),
                                features[i].norm() * features[self.idx_unlabel].norm(dim=1))
                _, rank = sim.topk(K)
                indices = self.idx_unlabel[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
            for i in self.idx_unlabel:
                sim = torch.div(torch.matmul(features[i], features[idx_train].T),
                                features[i].norm() * features[idx_train].norm(dim=1))
                _, rank = sim.topk(K)
                indices = idx_train[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        edge_index = list(edge_index.T)
        poten_edges = set([tuple(t) for t in poten_edges]) - set([tuple(t) for t in edge_index])
        poten_edges = [list(s) for s in poten_edges]
        poten_edges = torch.as_tensor(poten_edges).T.to(self.device)

        return poten_edges

    def get_pseudo_label(self, pred0, pred1):
        filter_condition = ((pred0.max(dim=1)[1][self.idx_unlabel] == pred1.max(dim=1)[1][self.idx_unlabel]) &
                            (pred0.max(dim=1)[0][self.idx_unlabel] * pred1.max(dim=1)[0][
                                self.idx_unlabel] > self.conf.model['th'] ** 2))
        idx_add = self.idx_unlabel[filter_condition]

        return idx_add.detach()
