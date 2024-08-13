from predictor.Base_Predictor import Predictor
from predictor.module.Coteaching import Coteaching
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
import nni


class coteaching_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = Coteaching(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                                n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                                norm_info=conf.model['norm_info'],
                                act=conf.model['act'], input_layer=conf.model['input_layer'],
                                output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.ek = conf.model["ek"]
        self.noise_rate = conf.model["noise_rate"]

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            feature, adj = self.feats, self.adj

            # forward and backward
            output1, output2 = self.model(feature, adj)

            pred_1 = output1[self.train_mask].max(1)[1]
            pred_2 = output2[self.train_mask].max(1)[1]

            disagree = (pred_1 != pred_2).cpu().numpy()
            idx_update = self.train_mask[disagree]

            # if len(idx_update) == 0: break

            k = int((1 - min(epoch * self.noise_rate / self.ek, self.noise_rate)) * len(idx_update))
            loss_1 = F.cross_entropy(output1[idx_update], self.noisy_label[self.train_mask][disagree], reduction='none')
            loss_2 = F.cross_entropy(output2[idx_update], self.noisy_label[self.train_mask][disagree], reduction='none')

            _, topk_1 = torch.topk(loss_1, k, largest=False)
            _, topk_2 = torch.topk(loss_2, k, largest=False)

            loss_train = loss_1[topk_2].mean() + loss_2[topk_1].mean()
            loss_train.backward()

            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output1[self.train_mask].detach().cpu().numpy())


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
                loss_test, acc_test = self.test(self.test_mask)
                nni.report_intermediate_result(acc_test)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        feature, adj = self.feats, self.adj
        with torch.no_grad():
            output1, output2 = self.model(feature, adj)
        logits = output1[mask]
        loss = self.loss_fn(logits, label)
        return loss, self.metric(label.cpu().numpy(), logits.detach().cpu().numpy())
