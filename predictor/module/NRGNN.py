import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.nn import GCNConv
from predictor.module.GNNs import GCN

# %%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, conf):
        super(EstimateAdj, self).__init__()

        # self.estimator = GCN(n_feat, conf.model['edge_hidden'], conf.model['edge_hidden'], dropout=0.0)
        self.estimator = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['edge_hidden'], out_channels=conf.model['edge_hidden'],
                             n_layers=conf.model['n_layer'], dropout=0.0,
                             norm_info=conf.model['norm_info'],
                             act=conf.model['act'], input_layer=conf.model['input_layer'],
                             output_layer=conf.model['output_layer'])
        self.conf = conf
        self.representations = 0

    def forward(self, edge_index, features):
        representations = self.estimator(features, edge_index)
        rec_loss = self.reconstruct_loss(edge_index, representations)
        return representations, rec_loss

    def get_estimated_weigths(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)

        estimated_weights = F.relu(output)
        estimated_weights[estimated_weights < self.conf.model['t_small']] = 0.0

        return estimated_weights

    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=self.conf.model['n_n'] * num_nodes)
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

'''
class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_bias=True, with_relu=True,
                 self_loop=True):

        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GCNConv(nfeat, nhid, bias=with_bias, add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias, add_self_loops=self_loop)
        self.dropout = dropout
        self.with_relu = with_relu

    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            x = F.relu(self.gc1(x, edge_index, edge_weight))
        else:
            x = self.gc1(x, edge_index, edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        return x

'''
