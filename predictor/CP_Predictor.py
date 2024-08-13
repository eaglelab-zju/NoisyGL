from predictor.Base_Predictor import Predictor
from predictor.module.CP import CPGCN
import time
import torch
from copy import deepcopy
from torch_geometric.nn.models import Node2Vec
from sklearn.cluster import KMeans
import nni


class cp_Predictor(Predictor):

    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = CPGCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                           n_clusters=conf.model['n_cluster'], n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                           norm_info=conf.model['norm_info'], act=conf.model['act'], input_layer=conf.model['input_layer'],
                           output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.embedding = self.get_graph_emb(self.adj, self.conf)
        self.community_labels = self.get_communities(conf, self.embedding)

    def get_graph_emb(self, adj, conf):
        edge_index = adj.indices()
        encoder = Node2Vec(edge_index,
                           embedding_dim=conf.model['emb_dim'],
                           walks_per_node=conf.model['emb_walks_per_node'],
                           walk_length=conf.model['emb_walk_length'],
                           context_size=conf.model['emb_context_size'],
                           p=conf.model['emb_p'],
                           q=conf.model['emb_q'],
                           num_negative_samples=conf.model['emb_num_negative_samples'],
                           sparse=True).to(self.device)
        optimizer = torch.optim.SparseAdam(encoder.parameters(), lr=conf.training['lr'])
        loader = encoder.loader(batch_size=conf.model['emb_batch_size'],
                                shuffle=True, num_workers=0)

        if self.conf.training['debug']:
            print("Node embedding start")
        for epoch in range(conf.training['emb_epochs']):
            # Training encoder
            encoder.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = encoder.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss = total_loss / len(loader)

            # Evaluating encoder
            encoder.eval()
            emb = encoder()
            acc = encoder.test(emb[self.train_mask], self.clean_label[self.train_mask],
                               emb[self.test_mask], self.clean_label[self.test_mask], max_iter=150)
            if self.conf.training['debug']:
                print(f'Embedding epoch:{epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        emb = encoder().data
        return emb

    def get_communities(self, conf, embedding):
        kmeans = KMeans(conf.model['n_cluster'])
        kmeans.fit(embedding.cpu().numpy())
        community_labels = torch.tensor(kmeans.labels_, dtype=torch.int64).to(self.device)
        if self.conf.training['debug']:
            print("Community detection finished")
        return community_labels

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj

            # forward and backward
            pred, pred_cluster = self.model(features, adj)
            loss_pred = self.loss_fn(pred[self.train_mask], self.noisy_label[self.train_mask])
            loss_cluster = self.loss_fn(pred_cluster[self.train_mask],
                                        self.community_labels[self.train_mask])

            loss_train = loss_pred + self.conf.model['lam'] * loss_cluster
            loss_train.backward()
            self.optim.step()

            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    pred[self.train_mask].detach().cpu().numpy())

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

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            output, _ = self.model(features, adj)
        logits = output[mask]
        loss = self.loss_fn(logits, label)
        return loss, self.metric(label.cpu().numpy(), logits.detach().cpu().numpy())
