import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy


class RNCGLN_model(nn.Module):
    def __init__(self, random_aug_feature, Trans_layer_num, trans_dim, n_heads, dropout_att, ft_size, n_class):
        super(RNCGLN_model, self).__init__()

        self.dropout = random_aug_feature
        self.Trans_layer_num = Trans_layer_num
        self.layers = get_clones(EncoderLayer(trans_dim, n_heads, dropout_att), Trans_layer_num)
        self.norm_input = Norm(ft_size)
        self.MLPfirst = nn.Linear(ft_size, trans_dim)
        self.MLPlast = nn.Linear(trans_dim, n_class)
        self.norm_layer = Norm(trans_dim)

    def forward(self, x_input):
        x_input = self.norm_input(x_input)
        x = self.MLPfirst(x_input)
        x = F.dropout(x, self.dropout, training=self.training)
        x_dis = get_feature_dis(self.norm_layer(x))
        for i in range(self.Trans_layer_num):
            x = self.layers[i](x)

        CONN_INDEX = F.relu(self.MLPlast(x))

        return F.softmax(CONN_INDEX, dim=1), x_dis


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.effectattn = EfficientAttention(in_channels=d_model, key_channels=d_model, head_count=heads, value_channels=d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        x_pre = self.effectattn(x2)
        x = x + x_pre
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(in_channels, key_channels)
        self.queries = nn.Linear(in_channels, key_channels)
        self.values = nn.Linear(in_channels, value_channels)
        self.reprojection = nn.Linear(key_channels, key_channels)

    def forward(self, input_):
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels], dim=0)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels], dim=1)
            value = values[:,i * head_value_channels: (i + 1) * head_value_channels]
            context = key.transpose(0, 1) @ value
            attended_value = query @ context
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        return attention


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = (1-mask) * x_dis*(x_sum**(-1))
    # x_dis = (1-mask) * x_dis
    return x_dis


def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj
    return adj_label


def Ncontrast(x_dis, adj_label, tau = 1, train_index_sort=None):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum_mid = torch.sum(x_dis, 1)
    x_dis_sum_pos_mid = torch.sum(x_dis*adj_label, 1)
    x_dis_sum = x_dis_sum_mid[train_index_sort]
    x_dis_sum_pos = x_dis_sum_pos_mid[train_index_sort]
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss


def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A
