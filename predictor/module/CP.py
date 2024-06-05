import torch.nn as nn
from predictor.module.GNNs import GCN


class CPGCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, n_clusters, n_layers=5, dropout=0.5, norm_info=None,
                 act='F.relu', input_layer=False, output_layer=False, bias=True):

        super(CPGCN, self).__init__()

        self.nfeat = in_channels
        self.hidden_sizes = [hidden_channels]
        self.nclass = out_channels

        self.dropout = dropout

        self.GCN = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels,
                       n_layers=n_layers, dropout=dropout, norm_info=norm_info,
                       act=act, input_layer=input_layer,
                       output_layer=output_layer, bias=bias)
        self.fc1 = nn.Linear(hidden_channels, out_channels)
        self.fc2 = nn.Linear(hidden_channels, n_clusters)

    def forward(self, x, adj):
        x = self.GCN(x, adj)
        pred = self.fc1(x)
        pred_cluster = self.fc2(x)
        return pred, pred_cluster



