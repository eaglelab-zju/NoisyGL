from predictor.Base_Predictor import Predictor
from predictor.module.RNCGLN import RNCGLN_model, get_A_r, Ncontrast
import torch
import time
from copy import deepcopy
import nni


class rncgln_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = RNCGLN_model(random_aug_feature=conf.model['random_aug_feature'],
                                  Trans_layer_num=conf.model['Trans_layer_num'],
                                  trans_dim=conf.model['trans_dim'], n_heads=conf.model['n_heads'],
                                  dropout_att=conf.model['dropout_att'],
                                  ft_size=conf.model['n_feat'], n_class=conf.model['n_classes']).to(self.device)
        self.optim = torch.optim.Adam(
                                list(self.model.parameters()),
                                lr=self.conf.training['lr'],
                                weight_decay=self.conf.training['weight_decay'])
        self.tau = conf.model['tau']
        self.order = conf.model['order']
        self.r1 = conf.model['r1']
        self.IsGraNoise = conf.model['IsGraNoise']
        self.SamSe = conf.model['SamSe']
        self.P_sel_onehot = conf.model['P_sel_onehot']
        self.P_sel = conf.model['P_sel']
        self.P_gra_sel = conf.model['P_gra_sel']

    def train(self):
        features, adj = self.feats, self.adj
        ones = torch.sparse.torch.eye(self.n_classes).to(self.device)
        self.labels_oneHot = ones.index_select(0, self.noisy_label)
        train_all_pos_bool = torch.ones(self.noisy_label.size(0))
        train_unsel = torch.cat((torch.tensor(self.val_mask), torch.tensor(self.test_mask)), dim=0).to(self.noisy_label.device)
        train_all_pos_bool[train_unsel] = 0
        train_all_pos = train_all_pos_bool.to(self.device)
        adj_label = get_A_r(adj, self.order)
        PP = 0
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            embeds_tra, x_dis = self.model(features)
            loss_cla = self.loss_fn(embeds_tra[self.train_mask], self.labels_oneHot[self.train_mask])
            max_va_pos = embeds_tra.max(1)[0]
            max_va_pos_index = max_va_pos >= PP
            loss_Ncontrast = Ncontrast(x_dis, adj_label, tau=self.tau, train_index_sort=max_va_pos_index)
            loss_train = loss_cla + self.r1 * loss_Ncontrast
            loss_train.backward()
            self.optim.step()

            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    embeds_tra[self.train_mask].detach().cpu().numpy())

            loss_val, acc_val = self.evaluate(self.noisy_label[self.val_mask], self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)

            if flag:
                improve = '*'
                better_inex = 1
                self.total_time = time.time() - self.start_time
                if epoch > self.conf.training['warmup_num']:
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.weights = deepcopy(self.model.state_dict())

                    embeds, _ = self.model(features)
                    pre_value_max, pre_index_max = embeds.max(1)
                    self.labels_oneHot = embeds.detach().clone()
                    Y_zero = torch.zeros_like(self.labels_oneHot)
                    Y_zero.scatter_(1, pre_index_max.unsqueeze(1), 1)
                    self.labels_oneHot[pre_value_max >= self.P_sel_onehot] = Y_zero[
                        pre_value_max >= self.P_sel_onehot]
                    pre_ind_min = pre_value_max >= self.P_sel
                    train_mask = pre_ind_min.float() + train_all_pos.float() == 2
                    # check1 = torch.nonzero(pre_ind_min.float())
                    # check2 = torch.nonzero(train_all_pos.float())
                    train_mask = torch.nonzero(train_mask).squeeze().cpu().numpy()
                    self.train_mask = train_mask
                    ## update some hype-parameters
                    if not self.IsGraNoise:
                        better_inex = 0
                    if self.SamSe:
                        PP = self.conf.model['P_sel_onehot']
                    if self.conf.training['debug']:
                        print("\n --> A new loop after label update")

                if epoch > self.conf.training['warmup_num'] and better_inex:
                    ## update GRAPH
                    x_dis_mid = x_dis.detach().clone()
                    x_dis_mid = x_dis_mid * adj_label
                    val_, pos_ = torch.topk(x_dis_mid, int(x_dis_mid.size(0) * (1 - self.P_gra_sel)))
                    # del adj_label
                    adj_label = torch.where(x_dis_mid > val_[:, -1].unsqueeze(1), x_dis_mid, 0)
                    if self.SamSe:
                        PP = self.P_sel_onehot
                    if self.conf.training['debug']:
                        print("\n --> A new loop after graph update")

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
        features, adj = self.feats, self.adj
        with torch.no_grad():
            output, _ = self.model(features)
        logits = output[mask]
        loss = self.loss_fn(logits, label)
        return loss, self.metric(label.cpu().numpy(), logits.detach().cpu().numpy())
