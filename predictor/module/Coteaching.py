import torch.nn as nn
from predictor.module.GNNs import GCN


class Coteaching(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5, norm_info=None,
                 act='F.relu', input_layer=False, output_layer=False, bias=True):

        super(Coteaching, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.dropout = dropout
        self.GCN1 = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                        n_layers=n_layers, dropout=dropout, norm_info=norm_info,
                        act=act, input_layer=input_layer,
                        output_layer=output_layer, bias=bias)
        self.GCN2 = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                        n_layers=n_layers, dropout=dropout, norm_info=norm_info,
                        act=act, input_layer=input_layer,
                        output_layer=output_layer, bias=bias)

    def forward(self, feature, adj):
        return self.GCN1(feature, adj), self.GCN2(feature, adj)

