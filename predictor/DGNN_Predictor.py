from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GIN
from predictor.module.DGNN import estimate_C, backward_correction
import time
import torch
from copy import deepcopy


class dgnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        '''
        This sets module and other members, which is overwritten for each method.
        '''
        self.pre_model = GIN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                             out_channels=conf.model['n_classes'],
                             n_layers=conf.model['n_layer'], mlp_layers=conf.model['mlp_layers'],
                             dropout=conf.model['dropout'],
                             train_eps=conf.model['train_eps']).to(self.device)
        self.pre_optim = torch.optim.Adam(self.pre_model.parameters(), lr=conf.training['lr'],
                                          weight_decay=conf.training['weight_decay'])

        self.model = GIN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], mlp_layers=conf.model['mlp_layers'],
                         dropout=conf.model['dropout'],
                         train_eps=conf.model['train_eps']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=conf.training['lr'],
                                      weight_decay=conf.training['weight_decay'])

    def train(self):
        '''
        This is the common learning procedure, which is overwritten for special learning procedure.

        Parameters
        ----------
        None

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        '''
        best_pre_acc = 0
        for pre_epoch in range(self.conf.training['n_pre_epochs']):
            self.pre_model.train()
            self.pre_optim.zero_grad()
            features, adj = self.feats, self.adj

            # forward and backward
            output = self.pre_model(features, adj)

            loss_train = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            if acc_train > best_pre_acc:
                best_pre_acc = acc_train
                pre_weight = self.pre_model.state_dict()

            loss_train.backward()
            self.pre_optim.step()
            if self.conf.training['debug']:
                print("pre_Epoch {:05d} | pre_Loss(train) {:.4f} | pre_Acc(train) {:.4f} | ".format(
                        pre_epoch + 1, loss_train.item(), acc_train))
        self.pre_model.load_state_dict(pre_weight)
        C = estimate_C(model=self.pre_model, x=features, adj=adj, n_classes=self.n_classes)

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj

            # forward and backward
            output = self.model(features, adj)


            loss_train = backward_correction(output[self.train_mask], self.noisy_label[self.train_mask], C)
            # loss_train = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
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
