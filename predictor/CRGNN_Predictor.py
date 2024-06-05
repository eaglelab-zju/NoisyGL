from predictor.Base_Predictor import Predictor
from predictor.module.CRGNN import *
import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, mask_feature


class crgnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = CRGNN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                           out_channels=conf.model['n_hidden'],).to(self.device)
        self.proj_head = ProjectionHead(in_channels=conf.model['n_hidden'], out_channels=conf.model['n_hidden']).to(self.device)
        self.class_head = ClassificationHead(in_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes']).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.proj_head.parameters()) +
                                      list(self.class_head.parameters()), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.T = 2
        self.tau = 0.5
        self.p = 0.8
        self.alpha = 0.2

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.proj_head.train()
            self.class_head.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.edge_index

            # forward and backward
            output = self.model(features, adj)
            edge_index1, _ = dropout_adj(adj, p=0.3)
            edge_index2, _ = dropout_adj(adj, p=0.3)
            x1 = mask_feature(features, p=0.3)[0]
            x2 = mask_feature(features, p=0.3)[0]

            # Extract representations
            h1 = self.model(x1, edge_index1)
            h2 = self.model(x2, edge_index2)

            # Project to contrast space
            z1 = self.proj_head(h1)
            z2 = self.proj_head(h2)

            # Calculate contrastive loss
            loss_con = contrastive_loss(z1, z2, self.tau)

            # Project to classification space
            p1 = self.class_head(h1)
            p2 = self.class_head(h2)

            # Compute pseudo-labels and dynamic cross-entropy loss
            loss_sup = dynamic_cross_entropy_loss(p1[self.train_mask], p2[self.train_mask], self.noisy_label[self.train_mask])

            # Compute similarity matrices
            # zm = torch.exp(F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=1) / self.T).mean(dim=1)
            # pm = torch.exp(F.cosine_similarity(p1.unsqueeze(1), p2.unsqueeze(0), dim=1) / self.T).mean(dim=1)

            # Apply thresholding in classification space
            # pm = torch.where(pm > self.p, pm, torch.zeros_like(pm))
            # Calculate cross-space consistency loss
            # loss_ccon = cross_space_consistency_loss(zm, pm)

            # Total loss+ beta * loss_ccon
            loss_train = self.alpha * loss_con + loss_sup
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    self.class_head(output)[self.train_mask].detach().cpu().numpy())
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
        self.model.load_state_dict(self.weights)
        label = self.clean_label[self.test_mask]
        self.model.eval()
        self.class_head.eval()
        self.proj_head.eval()
        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask):
        '''
        This is the common evaluation procedure, which is overwritten for special evaluation procedure.

        Parameters
        ----------
        label : torch.tensor
        mask: torch.tensor

        Returns
        -------
        loss : float
            Evaluation loss.
        metric : float
            Evaluation metric.
        '''
        self.model.eval()
        self.class_head.eval()
        self.proj_head.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            feats = self.model(features, adj)
            output = self.class_head(feats)
        logits = output[mask]
        loss = self.loss_fn(logits, label)
        return loss, self.metric(label.cpu().numpy(), logits.detach().cpu().numpy())

    def test(self, mask):
        '''
        This is the common test procedure, which is overwritten for special test procedure.

        Returns
        -------
        loss : float
            Test loss.
        metric : float
            Test metric.
        '''
        self.model.load_state_dict(self.weights)
        label = self.clean_label[mask]
        return self.evaluate(label, mask)
