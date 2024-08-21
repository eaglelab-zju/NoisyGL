from predictor.Base_Predictor import Predictor
from predictor.module.CRGNN import *
import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, mask_feature
import nni


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
        self.T = conf.model["T"]
        self.tau = conf.model["tau"]
        self.p = conf.model["p"]
        self.alpha = conf.model["alpha"]
        self.beta = conf.model["beta"]

    def get_prediction(self, features, adj, label=None, mask=None):
        adj = self.edge_index
        output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
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
            loss_sup = dynamic_cross_entropy_loss(p1[mask], p2[mask], label[mask])

            # Compute similarity matrices
            # zm = torch.exp(F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=1) / self.T).mean(dim=1)
            # pm = torch.exp(F.cosine_similarity(p1.unsqueeze(1), p2.unsqueeze(0), dim=1) / self.T).mean(dim=1)

            # Apply thresholding in classification space
            # pm = torch.where(pm > self.p, pm, torch.zeros_like(pm))
            # Calculate cross-space consistency loss
            # loss_ccon = cross_space_consistency_loss(zm, pm)

            # Total loss+ beta * loss_ccon
            loss = self.alpha * loss_con + loss_sup
            acc = self.metric(label[mask].cpu().numpy(), self.class_head(output)[mask].detach().cpu().numpy())
        return output, loss, acc



    def train(self):
        t0 = time.time()
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            self.model.train()
            self.proj_head.train()
            self.class_head.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, self.train_mask)
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

    def evaluate(self, label, mask):
        self.model.eval()
        self.class_head.eval()
        self.proj_head.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            feats = self.model(features, adj)
            output = self.class_head(feats)
        loss = self.loss_fn(output[mask], label[mask])
        acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return loss, acc
    #
    # def test(self, mask):
    #     '''
    #     This is the common test procedure, which is overwritten for special test procedure.
    #
    #     Returns
    #     -------
    #     loss : float
    #         Test loss.
    #     metric : float
    #         Test metric.
    #     '''
    #     self.model.load_state_dict(self.weights)
    #     label = self.clean_label[mask]
    #     return self.evaluate(label, mask)
