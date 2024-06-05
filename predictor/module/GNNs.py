import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 dropout=0.5, use_bn=True):
        super(MLP, self).__init__()
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if n_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, mlp_layers=1, dropout=0.5, train_eps=True):
        super(GIN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.train_eps = train_eps

        self.convs = nn.ModuleList()
        if n_layers == 1:
            self.convs.append(GINConv(MLP(in_channels, hidden_channels, out_channels, mlp_layers, dropout), train_eps=train_eps))
        else:
            self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels, mlp_layers, dropout), train_eps=train_eps))
            for layer in range(self.n_layers - 2):
                self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels, mlp_layers, dropout), train_eps=train_eps))
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, out_channels, mlp_layers, dropout), train_eps=train_eps))

    def forward(self, x, adj):
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, adj)
            x = F.relu(x)
        x = self.convs[-1](x, adj)
        return x


class GCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=5, dropout=0.5, norm_info=None,
                 act='F.relu', input_layer=False, output_layer=False, bias=True, add_self_loops=True):

        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.dropout = dropout
        if norm_info is None:
            norm_info = {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm = eval('nn.' + norm_info['norm_type'])
        self.act = eval(act)
        if input_layer:
            self.input_linear = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        if output_layer:
            self.output_linear = nn.Linear(in_features=hidden_channels, out_features=out_channels)
            self.output_normalization = self.norm_type(hidden_channels)
        self.convs = nn.ModuleList()
        if self.is_norm:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = in_channels
            else:
                in_hidden = hidden_channels
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = out_channels
            else:
                out_hidden = hidden_channels
            self.convs.append(GCNConv(in_hidden, out_hidden, bias=bias, add_self_loops=add_self_loops))
            if self.is_norm:
                self.norms.append(self.norm_type(in_hidden))
        self.convs[-1].last_layer = True

    def forward(self, x, adj):
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)

        for i, layer in enumerate(self.convs):
            if self.is_norm:
                x_res = self.norms[i](x)
                x_res = layer(x_res, adj)
                x = x + x_res
            else:
                x = layer(x, adj)
            if i < self.n_layers - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.output_layer:
            x = self.output_normalization(x)
            x = self.output_linear(x).squeeze(1)

        return x.squeeze(1)



