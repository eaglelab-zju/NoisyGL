import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
import torch_geometric.utils as utils
from predictor.module.GNNs import GCN


def kl_loss_compute(pred, soft_targets, reduce=True, tempature=1):
    pred = pred / tempature
    soft_targets = soft_targets / tempature
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


class LabeledDividedLoss(nn.Module):
    def __init__(self, conf):
        super(LabeledDividedLoss, self).__init__()
        self.epochs = conf.training['n_epochs']
        self.increment = 0.5 / self.epochs
        self.decay_w = conf.model['decay_w']

    def forward(self, y_1, y_2, t, co_lambda=0.1, epoch=-1):
        loss_pick_1 = F.cross_entropy(y_1, t, reduce=False)
        loss_pick_2 = F.cross_entropy(y_2, t, reduce=False)
        loss_pick = loss_pick_1 + loss_pick_2

        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        forget_rate = self.increment * epoch
        remember_rate = 1 - forget_rate
        mean_v = loss_sorted.mean()
        idx_small = torch.where(loss_sorted < mean_v)[0]

        remember_rate_small = idx_small.shape[0] / t.shape[0]

        remember_rate = max(remember_rate, remember_rate_small)
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]

        loss_clean = torch.sum(loss_pick[ind_update]) / y_1.shape[0]
        ind_all = torch.arange(1, t.shape[0]).long()
        ind_update_1 = torch.LongTensor(
            list(set(ind_all.detach().cpu().numpy()) - set(ind_update.detach().cpu().numpy()))).to(ind_update.device)
        p_1 = F.softmax(y_1, dim=-1)
        p_2 = F.softmax(y_2, dim=-1)

        filter_condition = ((y_1.max(dim=1)[1][ind_update_1] != t[ind_update_1]) &
                            (y_1.max(dim=1)[1][ind_update_1] == y_2.max(dim=1)[1][ind_update_1]) &
                            (p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1] > (
                                    1 - (1 - min(0.5, 1 / y_1.shape[0])) * epoch / self.epochs)))
        dc_idx = ind_update_1[filter_condition]

        adpative_weight = (p_1.max(dim=1)[0][dc_idx] * p_2.max(dim=1)[0][dc_idx]) ** (
                0.5 - 0.5 * epoch / self.epochs)
        loss_dc = adpative_weight * (F.cross_entropy(y_1[dc_idx], y_1.max(dim=1)[1][dc_idx], reduce=False) + \
                                     F.cross_entropy(y_2[dc_idx], y_1.max(dim=1)[1][dc_idx], reduce=False))
        loss_dc = loss_dc.sum() / y_1.shape[0]

        remain_idx = torch.LongTensor(
            list(set(ind_update_1.detach().cpu().numpy()) - set(dc_idx.detach().cpu().numpy())))

        loss1 = torch.sum(loss_pick[remain_idx]) / y_1.shape[0]
        decay_w = self.decay_w

        inter_view_loss = kl_loss_compute(y_1, y_2).mean() + kl_loss_compute(y_2, y_1).mean()

        return loss_clean + loss_dc + decay_w * loss1 + co_lambda * inter_view_loss


class PseudoLoss(nn.Module):
    def __init__(self):
        super(PseudoLoss, self).__init__()

    def forward(self, y_1, y_2, idx_add, co_lambda=0.1):
        pseudo_label = y_1.max(dim=1)[1]
        loss_pick_1 = F.cross_entropy(y_1[idx_add], pseudo_label[idx_add], reduce=False)
        loss_pick_2 = F.cross_entropy(y_2[idx_add], pseudo_label[idx_add], reduce=False)
        loss_pick = loss_pick_1.mean() + loss_pick_2.mean()
        inter_view_loss = kl_loss_compute(y_1[idx_add], y_2[idx_add]).mean() + kl_loss_compute(y_2[idx_add],
                                                                                               y_1[idx_add]).mean()
        loss = torch.mean(loss_pick) + co_lambda * inter_view_loss

        return loss


class IntraviewReg(nn.Module):
    def __init__(self, device):
        super(IntraviewReg, self).__init__()
        self.device = device

    def index_to_mask(self, index, size=None):
        index = index.view(-1)
        size = int(index.max()) + 1 if size is None else size
        mask = index.new_zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

    def bipartite_subgraph(self, subset, edge_index, max_size):
        subset = (self.index_to_mask(subset[0], size=max_size), self.index_to_mask(subset[1], size=max_size))
        node_mask = subset
        edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
        return torch.where(edge_mask == True)[0]

    def neighbor_cons(self, y_1, y_2, edge_index, edge_weight, idx):
        if idx.shape[0] == 0:
            return torch.Tensor([0]).to(self.device)
        weighted_adj = utils.to_scipy_sparse_matrix(edge_index, edge_weight.detach())
        colsum = np.array(weighted_adj.sum(0))
        r_inv = np.power(colsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        norm_adj = weighted_adj.dot(r_mat_inv)

        norm_idx, norm_weight = utils.from_scipy_sparse_matrix(norm_adj)
        norm_idx, norm_weight = norm_idx.to(self.device), norm_weight.to(self.device)
        idx_all = torch.arange(0, y_1.shape[0]).to(self.device)

        filter_idx = self.bipartite_subgraph((idx_all, idx), norm_idx.to(self.device), max_size=int(y_1.shape[0]))
        edge_index, edge_weight = norm_idx[:, filter_idx], norm_weight[filter_idx]
        edge_index, edge_weight = edge_index.to(self.device), edge_weight.to(self.device)

        intra_view_loss = (edge_weight * kl_loss_compute(y_1[edge_index[1]], y_1[edge_index[0]].detach())).sum() + \
                          (edge_weight * kl_loss_compute(y_2[edge_index[1]], y_2[edge_index[0]].detach())).sum()
        intra_view_loss = intra_view_loss / idx.shape[0]
        return intra_view_loss

    def forward(self, y_1, y_2, idx_label, edge_index, edge_weight):
        neighbor_kl_loss = self.neighbor_cons(y_1, y_2, edge_index, edge_weight, idx_label)
        return neighbor_kl_loss


class EstimateAdj(nn.Module):

    def __init__(self, conf):
        super(EstimateAdj, self).__init__()
        '''
        self.estimator = GCNEncoder(conf.model['n_feat'], conf.model['edge_hidden'], conf.model['edge_hidden'], dropout=0.0)
        '''
        self.estimator = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['edge_hidden'], out_channels=conf.model['edge_hidden'],
                             n_layers=conf.model['n_layer'], dropout=0.0,
                             norm_info=conf.model['norm_info'],
                             act=conf.model['act'], input_layer=conf.model['input_layer'],
                             output_layer=conf.model['output_layer'])
        self.conf = conf
        self.representations = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, adj):
        representations = self.estimator(features, adj)
        representations = F.normalize(representations, dim=-1)
        rec_loss = self.reconstruct_loss(adj.indices(), representations)
        return representations, rec_loss

    def get_estimated_weigths(self, edge_index, representations, origin_w=None):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)
        estimated_weights = F.relu(output)
        if estimated_weights.shape[0] != 0:
            estimated_weights[estimated_weights < self.conf.model['tau']] = 0
            if origin_w != None:
                estimated_weights = origin_w + estimated_weights * (1 - origin_w)

        return estimated_weights, None

    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes,
                                        num_neg_samples=self.conf.model['n_neg'] * num_nodes)

        randn = randn[:, randn[0] < randn[1]]
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0, neg1), dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0, pos1), dim=1)
        rec_loss = (F.mse_loss(neg, torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                   * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss


class Dual_GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_layer, dropout, norm_info, act, input_layer, output_layer, n_nodes):
        super(Dual_GCN, self).__init__()
        self.n_nodes = n_nodes
        self.GCN_1 = GCN(in_channels=n_feat, hidden_channels=n_hid, out_channels=n_class, n_layers=n_layer, dropout=dropout,
                         norm_info=norm_info,
                         act=act, input_layer=input_layer, output_layer=output_layer)
        self.GCN_2 = GCN(in_channels=n_feat, hidden_channels=n_hid, out_channels=n_class, n_layers=n_layer, dropout=dropout,
                         norm_info=norm_info,
                         act=act, input_layer=input_layer, output_layer=output_layer)

    def forward(self, x, edge_index, edge_weight):
        reformed_adj = torch.sparse_coo_tensor(edge_index, edge_weight, [self.n_nodes, self.n_nodes])
        x1 = self.GCN_1(x, reformed_adj)
        x2 = self.GCN_2(x, reformed_adj)
        return x1, x2


class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True,
                 self_loop=True):

        super(GCNEncoder, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GCNConv(nfeat, nhid, bias=with_bias, add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias, add_self_loops=self_loop)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def forward(self, x, adj):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

'''
class Dual_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5,  with_relu=True, with_bias=True,
                 self_loop=True):

        super(Dual_GCN, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1_1 = GCNConv(nfeat, nhid, bias=with_bias, add_self_loops=self_loop)
        self.gc2_1 = GCNConv(nhid, nclass, bias=with_bias, add_self_loops=self_loop)
        self.gc1_2 = GCNConv(nfeat, nhid, bias=with_bias, add_self_loops=self_loop)
        self.gc2_2 = GCNConv(nhid, nclass, bias=with_bias, add_self_loops=self_loop)

        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bias = with_bias

    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            x1 = F.relu(self.gc1_1(x, edge_index, edge_weight))
        else:
            x1 = self.gc1_1(x, edge_index, edge_weight)

        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2_1(x1, edge_index, edge_weight)

        if self.with_relu:
            x2 = F.relu(self.gc1_2(x, edge_index, edge_weight))
        else:
            x2 = self.gc1_2(x, edge_index, edge_weight)

        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = self.gc2_2(x2, edge_index, edge_weight)

        return x1,x2




'''