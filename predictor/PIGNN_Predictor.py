from predictor.Base_Predictor import Predictor
from predictor.module.PIGNN import GCN_pignn
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
import nni


class pignn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GCN_pignn(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                               out_channels=conf.model['n_classes'],
                               n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                               norm_info=conf.model['norm_info'],
                               act=conf.model['act'], input_layer=conf.model['input_layer'],
                               output_layer=conf.model['output_layer']).to(self.device)
        self.model_mi = GCN_pignn(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                                  out_channels=conf.model['n_classes'],
                                  n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                                  norm_info=conf.model['norm_info'],
                                  act=conf.model['act'], input_layer=conf.model['input_layer'],
                                  output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])
        self.optim_mi = torch.optim.Adam(
            list(self.model_mi.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.model_mi.train()
            self.optim.zero_grad()
            self.optim_mi.zero_grad()
            features, adj = self.feats, self.adj
            labels_context = adj.to_dense() + torch.eye(adj.shape[0]).to(adj.device)
            data_context = torch.ones(self.edge_index.shape[1])
            output, output_product = self.model(features, adj)
            out_mi, out_product_mi = self.model_mi(features, adj)
            loss_train = F.nll_loss(output[self.train_mask], self.noisy_label[self.train_mask])
            norm = self.n_nodes * self.n_nodes / float((self.n_nodes * self.n_nodes - data_context.shape[0]) * 2)
            pos_weight = torch.Tensor(
                [float(self.n_nodes * self.n_nodes - data_context.shape[0]) / data_context.shape[0]]).to(self.device)
            loss_mi = norm * F.binary_cross_entropy_with_logits(
                out_product_mi, labels_context, pos_weight=pos_weight)
            loss_mi.backward()
            self.optim_mi.step()
            mask = torch.zeros_like(out_product_mi).view(-1).to(self.device)
            pos_position = labels_context.view(-1).bool().to(self.device)
            neg_position = (1 - labels_context).view(-1).bool().to(self.device)

            mask[pos_position] = torch.sigmoid(out_product_mi).view(-1)[pos_position]
            mask[neg_position] = 1 - torch.sigmoid(out_product_mi).view(-1)[neg_position]
            mask = mask.view(labels_context.size(0), labels_context.size(1))

            if epoch > self.conf.training['start_epoch']:
                loss_context = norm * (F.binary_cross_entropy_with_logits(
                    output_product, labels_context, pos_weight=pos_weight, reduction='none') * mask.detach()).mean()
            else:
                loss_context = norm * F.binary_cross_entropy_with_logits(
                    output_product, labels_context, pos_weight=pos_weight)
            loss_train += loss_context
            loss_train.backward()
            self.optim.step()

            self.model.eval()

            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())

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
