from predictor.Base_Predictor import Predictor
from predictor.module.LCAT import GATv2Layer, GAT2v2Layer
import torch
import torch.nn.functional as F
import time
import nni
from copy import deepcopy


class lcat_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.dropout = conf.model["dropout"]
        if conf.model['module'] == 'gat_gcn_v2':
            self.model = GATv2Layer(in_channels=conf.model['n_hidden'], out_channels=conf.model['n_hidden'],
                                    mode='lcat', heads=conf.model['heads'],
                                    negative_slope=conf.model['negative_slope']).to(self.device)
            self.output_lin = torch.nn.Linear(in_features=conf.model['n_hidden'] * conf.model['heads'],
                                              out_features=conf.model['n_classes']).to(self.device)
        elif conf.model['module'] == 'gat_gcn2_v2':
            self.model = GAT2v2Layer(channels=conf.model['n_hidden'],
                                     mode='lcat', heads=conf.model['heads'], alpha=conf.model['alpha'],
                                     layer=conf.model["layer"], theta=conf.model["theta"],
                                     negative_slope=conf.model['negative_slope'],
                                     share_weights_score=True, share_weights_value=True,).to(self.device)
            self.output_lin = torch.nn.Linear(in_features=conf.model['n_hidden'],
                                              out_features=conf.model['n_classes']).to(self.device)
        else:
            print("invalid module " + conf.model["module"])
            exit(1)
        self.input_lin = torch.nn.Linear(in_features=conf.model['n_feat'], out_features=conf.model['n_hidden']).to(self.device)

        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.input_lin.parameters())
                                      + list(self.output_lin.parameters()), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            # forward and backward

            input_feats = self.input_lin(features)
            input_feats = F.dropout(input_feats, p=self.dropout)
            output = self.model(x=input_feats, edge_index=adj)
            output = F.dropout(output, p=self.dropout)
            output = self.output_lin(output)

            loss_train = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
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
                self.input_weight = deepcopy(self.input_lin.state_dict())
                self.output_weight = deepcopy(self.output_lin.state_dict())
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
        self.output_lin.eval()
        self.input_lin.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            input_feat = self.input_lin(features)
            output = self.model(input_feat, adj)
            output = self.output_lin(output)
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
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
            self.input_lin.load_state_dict(self.input_weight)
            self.output_lin.load_state_dict(self.output_weight)
        label = self.clean_label[mask]
        return self.evaluate(label, mask)
