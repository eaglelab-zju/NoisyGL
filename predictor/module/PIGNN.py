import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from predictor.module.GNNs import GCN


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, act=lambda x: x):
        super(InnerProductDecoder, self).__init__()
        # self.dropout = dropout
        self.act = act

    def forward(self, z):
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GCN_pignn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5, norm_info=None,
                 act='F.relu', input_layer=False, output_layer=False, bias=True):
        super(GCN_pignn, self).__init__()
        self.GCN = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                       n_layers=n_layers, dropout=dropout, norm_info=norm_info,
                       act=act, input_layer=input_layer,
                       output_layer=output_layer, bias=bias)
        self.dc = InnerProductDecoder()
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.GCN(x, adj)
        x_product = self.dc(x)
        return F.log_softmax(x, dim=1), x_product


