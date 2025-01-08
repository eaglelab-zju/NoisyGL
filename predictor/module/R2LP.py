import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import numpy as np
import scipy.sparse as sp

class MLP(nn.Module):
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, delta):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.fc3 = nn.Linear(nnodes, nhid)
        self.nclass = nclass
        self.dropout = dropout
        self.delta = delta

    def forward(self, x, adj):
        xX = F.dropout(x, self.dropout, training=self.training)
        xX = self.fc1(x)
        xA = self.fc3(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x

class R2LP(nn.Module):
    def __init__(self, nnodes, in_channels, hidden_channels, out_channels, dropout, alpha, alpha1, alpha2, alpha3, beta, gamma, delta, norm_func_id, norm_layers, orders, orders_func_id, device):
        super(R2LP, self).__init__()
        self.mlp = MLP(nnodes, in_channels, hidden_channels, out_channels, dropout, delta).to(device)
        self.nclass = out_channels
        self.dropout = dropout
        self.alpha = torch.tensor(alpha).to(device)
        self.alpha1 = torch.tensor(alpha1).to(device)
        self.alpha2 = torch.tensor(alpha2).to(device)
        self.alpha3 = torch.tensor(alpha3).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.delta = torch.tensor(delta).to(device)
        self.norm_layers = norm_layers
        self.orders = orders
        self.class_eye = torch.eye(out_channels).to(device)
        self.nodes_eye = torch.eye(nnodes).to(device)

        self.orders_weight = Parameter(
            torch.ones(orders, 1) / orders, requires_grad=True
        ).to(device)
        # use kaiming_normal to initialize the weight matrix in Orders3
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(out_channels, orders), requires_grad=True
        ).to(device)
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(orders, orders), requires_grad=True
        ).to(device)
        # use diag matirx to initialize the second norm layer
        self.diag_weight = Parameter(
            torch.ones(out_channels, 1) / out_channels, requires_grad=True
        ).to(device)
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()

        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def forward(self, x, adj, y_clean, y_unknown, if_lp):
        x = self.mlp(x, adj)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm_func1(x, h0, adj)

        if if_lp:
            y_predict = F.softmax(x)
            z, fl = self.norm_func2(x, h0, adj, y_clean, y_unknown, y_predict)
            return F.log_softmax(x, dim=1), z, y_predict, x, fl
        else:
            return F.log_softmax(x, dim=1)


    def norm_func1(self, x, h0, adj):
        # print('norm_func1 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res1 = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.linalg.pinv(coe2 * coe2 * self.class_eye + coe * res1)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res1)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
              self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0

        return res

    def norm_func2(self, x, h0, adj, y_clean, y_unknown, y_predict):
        # print('norm_func2 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res1 = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res1)

        # # calculate z
        xx = torch.mm(x, x.t())
        hx = torch.mm(h0, x.t())
        adj = adj.to_dense()
        adjk = adj
        a_sum = adjk * self.orders_weight[0]
        ress = torch.mm(inv, x.t())
        res = coe * self.nodes_eye - coe * coe *torch.mm(x, ress)

        alpha = self.alpha1
        miu = ( 1 -self.alpha1 ) *self.alpha2
        lamada = ( 1 -self.alpha1 ) *( 1 -self.alpha2 ) *self.alpha3
        delta = ( 1 -self.alpha1 ) *( 1 -self.alpha2 ) *( 1 -self.alpha3)

        res1 = coe1 * xx + self.beta * a_sum - self.gamma * coe1 * hx
        z = torch.mm(res1, res)
        y0 = delta * y_predict + miu * y_clean + lamada * y_unknown

        f1 = torch.mm(x.t(), y0)
        f2 = inv
        f3 = torch.mm(f2, f1)
        f4 = coe * y0 - coe * coe *torch.mm(x, f3)
        f5 = alpha * torch.mm(res1, f4)
        fl = y0 + f5
        return z, fl

    def order_func1(self, x, res, adj):
        # Orders1
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # Orders2
        tmp_orders = torch.spmm(adj, res)
        # print('tmp_orders', tmp_orders.shape)
        # print('orders_weight', self.orders_weight[0].shape)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        # Orders3
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        # orders_para = torch.mm(x, self.orders_weight_matrix)
        orders_para = torch.transpose(orders_para, 0, 1)
        tmp_orders = torch.spmm(adj, res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders


def Z_to_A(h_l, Z, eps_adj, func_Z):
    if func_Z == 1:
        A = Z
    else:
        smi = torch.cdist(h_l, h_l)
        A = torch.exp(-smi)
        B = torch.zeros((A.shape[0], A.shape[0])).cuda()
        C = torch.ones((A.shape[0], A.shape[0])).cuda()
        A = torch.where(A > eps_adj, C, B)
        A = calc_A_hat(A)
    return A

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    A = np.array(adj_matrix.cpu())
    D_vec = np.sum(A, axis=1)
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return torch.tensor(D_invsqrt_corr @ A @ D_invsqrt_corr).cuda()

def new_clean(F_t, pre_select, idx_clean, idx_unknown, labels, y_clean):
    labels = labels.clone()
    idx_unknown = idx_unknown.clone()
    idx_clean = idx_clean.clone()
    if idx_unknown.shape[0]:
        p = F_t
        topk_indices = torch.topk(p.max(dim=1)[0][idx_unknown], int(idx_unknown.shape[0 ] *pre_select), -1, True, False).indices
        idx_add = idx_unknown[topk_indices]
        pred_label = torch.argmax(p, dim=-1)
        labels[idx_add] = p.argmax(dim=-1)[idx_add]

        idx_unknown_list = set(idx_unknown.cpu().numpy())
        idx_add_list = set(idx_add.cpu().numpy())
        idx_unknown_list.difference_update(idx_add_list)
        idx_unknown_list = list(idx_unknown_list)

        # idx_unknown_list = list(np.array(idx_unknown.cpu()))
        # for _ in list(idx_add.cpu()):
        #     idx_unknown_list.remove(_)
        new_idx_unknown = torch.tensor(idx_unknown_list).to(idx_unknown.device)
        new_idx_clean = torch.cat((idx_clean ,idx_add))

    else:
        new_idx_unknown = idx_unknown
        new_idx_clean = idx_clean
        y_clean = y_clean
    return new_idx_unknown, new_idx_clean, labels

def label_correction(num_propagations, new_labels, Z, y_clean, y_unknown, idx_clean, idx_unknown, y_poredict, h_l,
                     lamada1, lamada2, lamada3, pre_select, eps_adj, func_Z, origi_labels):
    labels = new_labels
    adj = Z_to_A(h_l, Z, eps_adj, func_Z)

    alpha = lamada1
    miu = ( 1 -lamada1 ) *lamada2
    lamada = ( 1 -lamada1 ) *( 1 -lamada2 ) *lamada3
    delta = ( 1 -lamada1 ) *( 1 -lamada2 ) *( 1 -lamada3)

    F_0 = delta * y_poredict + miu * y_clean + lamada * y_unknown
    F_t = F_0.clone()
    for _ in range(num_propagations):
        F_t = (adj @ F_t) *alpha + delta * y_poredict + miu * y_clean + lamada * y_unknown
    F_t = F.softmax(F_t)

    new_idx_unknown, new_idx_clean, labels, y_clean = new_clean(F_t, pre_select, idx_clean, idx_unknown,
                                                                labels, y_clean, origi_labels)

    return F_t, new_idx_unknown, new_idx_clean, labels, y_clean

def num_cp(idx_clean, origi_labels):
    Cp = []
    for i in range(origi_labels.max().item() + 1):
        idx = (origi_labels[idx_clean] == i)
        ls = idx_clean[torch.nonzero(idx).squeeze()]
        Cp.append(ls)
    return Cp

def select(Cp, output):
    lamada_p = []
    for i in range(len(Cp)):
        miu_p = output[Cp[i] ,i].mean()
        sig_p = ((output[Cp[i] ,i ] -miu_p )**2).mean()
        lamada_i = miu_p + sig_p
        lamada_p.append(lamada_i)
    return lamada_p