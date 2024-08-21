import time
import torch
import torch.nn.functional as F
from copy import deepcopy
from predictor.Base_Predictor import Predictor
from predictor.module.CLNode import GCNNet, GCNClassifier, DifficultyMeasurer, training_scheduler
import copy
import nni


class clnode_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.lamb = conf.model['lambda']
        self.T = conf.model['T']
        self.scheduler = conf.model['scheduler']
        self.pre_model = GCNClassifier(n_feat=conf.model['n_feat'], n_hidden=conf.model['n_hidden'],
                                       n_class=conf.model['n_classes'], n_embedding=conf.model['n_emb'],
                                       dropout=conf.model['dropout']).to(self.device)
        self.model = GCNNet(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                            out_channels=conf.model['n_classes'],
                            n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                            norm_info=conf.model['norm_info'],
                            act=conf.model['act'], input_layer=conf.model['input_layer'],
                            output_layer=conf.model['output_layer']).to(self.device)
        self.difficultyMeasurer = DifficultyMeasurer(device=self.device, alpha=conf.model['alpha'])
        self.pre_optim = torch.optim.Adam(self.pre_model.parameters(),
                                          lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.conf.training['lr'], weight_decay=self.conf.training['weight_decay'])


    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = F.nll_loss(output[mask], self.noisy_label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        features = self.feats
        adj = self.adj
        pre_weight = None
        best_pre_val_acc = 0
        # print("Pre training")
        for pre_epoch in range(self.conf.training['n_pre_epochs']):
            self.pre_model.train()
            self.pre_optim.zero_grad()
            _, out = self.pre_model(features, adj)
            loss = F.nll_loss(out[self.train_mask], self.noisy_label[self.train_mask])
            loss.backward()
            self.pre_optim.step()

            self.pre_model.eval()
            _, out = self.pre_model(features, adj)
            _, pred = out.max(dim=1)
            correct = int(pred[self.val_mask].eq(self.noisy_label[self.val_mask]).sum().item())
            acc_val = correct / int(self.val_mask.shape[0])
            if acc_val > best_pre_val_acc:
                pre_weight = self.pre_model.state_dict()

        self.pre_model.eval()
        self.pre_model.load_state_dict(pre_weight)
        embedding, out = self.pre_model(features, adj)
        _, pred = out.max(dim=1)
        pred_label = copy.deepcopy(pred)
        pred_label[self.train_mask] = self.noisy_label[self.train_mask]

        # Test label is not available in the training process.
        # Here we replace the test label with constant integers when measuring difficulty,
        # which is different from the implementation provided by authors.

        # input_label = self.noisy_label.clone()
        input_label = ((torch.min(self.noisy_label) + 0) * torch.ones(self.noisy_label.shape, dtype=torch.int64).to(
            self.noisy_label.device))
        # input_label = torch.randint(low=torch.min(self.labels), high=torch.max(self.labels), size=self.labels.shape).to(self.labels.device)
        input_label[self.train_mask] = self.noisy_label[self.train_mask]
        input_label[self.val_mask] = self.noisy_label[self.val_mask]
        sorted_trainset_id, sorted_trainset_indices = self.difficultyMeasurer.sort_training_nodes(
            self.edge_index, self.train_mask, input_label, embedding)

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            size = training_scheduler(self.lamb, epoch, self.T, self.scheduler)
            batch_id = sorted_trainset_id[:int(size * sorted_trainset_id.shape[0])]
            batch_indices = sorted_trainset_indices[:int(size * sorted_trainset_indices.shape[0])]

            # output = self.model(features, adj)
            # loss_train = F.nll_loss(output[batch_id], self.noisy_label[batch_id])
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, batch_id)

            loss_train.backward()
            self.optim.step()

            # acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
            #                         output[self.train_mask].detach().cpu().numpy())

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
