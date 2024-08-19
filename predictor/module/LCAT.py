import math
from math import log
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import constant
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, degree
from torch_sparse import set_diag, SparseTensor
from torch_geometric.nn.dense.linear import Linear, Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GeneralGATLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            negative_slope: float = 0.2,
            add_self_loops: bool = True,
            heads: int = 3,
            bias: bool = True,
            mode: str = 'gcn',
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        assert mode in ['gcn', 'gat', 'cat', 'lcat', 'gcngat', 'gatcat']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        self.convolve = mode in ['gcn', 'cat', 'lcat', 'gatcat']

        self.learn_l1 = mode in ['lcat', 'gcngat']
        self.learn_l2 = mode in ['lcat', 'gatcat']

        self.mode = mode
        self.lmbda_ = None
        self.lmbda2_ = None

        if self.learn_l1:
            self.lmbda_ = nn.Parameter(torch.ones([]) * 0, requires_grad=True)
        if self.learn_l2:
            self.lmbda2_ = nn.Parameter(torch.ones([]) * 0, requires_grad=True)

        self.bias = 0.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels * self.heads))

    @property
    def lmbda(self):  # The one that controls GCN-->GATConv
        if self.learn_l1:
            return torch.sigmoid(10 ** self.lmbda_ - 6)
        else:
            if self.mode in ['gcn']:
                return 0.
            elif self.mode in ['gat', 'cat', 'gatcat']:
                return 1.0
            else:
                raise NotImplementedError

    @property
    def lmbda2(self):  # The one that controls GATConv-->GAT
        if self.learn_l2:
            return torch.sigmoid(10 ** (2.2 - self.lmbda2_) - 6)
        else:
            if self.mode in ['gcn', 'gat', 'gcngat']:
                return 0.0
            elif self.mode in ['cat']:
                return 1.
            else:
                raise NotImplementedError

    def reset_parameters(self):
        constant(self.lmbda_, math.log10(6))
        constant(self.lmbda2_, 2.2 - math.log10(6))
        constant(self.bias, 0.)

    def get_x_r(self, x):
        raise NotImplementedError

    def get_x_l(self, x):
        raise NotImplementedError

    def get_x_v(self, x):
        raise NotImplementedError

    def get_x_agg(self, x_l, x_r, edge_index, edge_weight):
        if isinstance(edge_index, Tensor):
            edge_index_no_neigh, edge_weight_no_neigh = remove_self_loops(edge_index, edge_weight)
        elif isinstance(edge_index, SparseTensor):
            edge_index_no_neigh = set_diag(edge_index, 0)
            edge_weight_no_neigh = None
        else:
            raise NotImplementedError

        aggr = self.aggr
        self.aggr = 'add'
        x_lr = torch.cat((x_l, x_r), dim=1)
        self.convolve = True
        x_neig_sum = self.propagate(edge_index_no_neigh, x=(x_lr, x_lr), size=None, edge_weight=edge_weight_no_neigh)
        self.aggr = aggr

        x_agg = self.lmbda * (x_lr + self.lmbda2 * x_neig_sum)

        row, col = edge_index_no_neigh.indices()
        counts = degree(col, x_agg.shape[0])
        # Divide by number of neighbors
        # i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        # if isinstance(edge_index, Tensor):
        #     counts = x_agg.new_zeros((x_agg.size(0),))
        #     a, b = edge_index_no_neigh[i].unique(return_counts=True)
        #     counts = counts.scatter_add(0, a, b.float())
        # elif isinstance(edge_index, SparseTensor):
        #     counts = edge_index_no_neigh.sum(dim=j)

        x_agg = x_agg / (1 + self.lmbda2 * counts.unsqueeze(-1).unsqueeze(-1))

        return x_agg

    def merge_heads(self, x):
        return x.flatten(start_dim=-2)

    def compute_score(self, x_i, x_j, index, ptr, size_i):
        raise NotImplementedError

    def fix_parameters(self, partial=False):
        raise NotImplementedError

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, size_target: int = None,
                edge_weight: OptTensor = None, return_attention_info: bool = False):

        assert isinstance(x, Tensor) and x.dim() == 2

        # We apply the linear layer before convolving to avoid numerical errors
        x_l = self.get_x_l(x)
        x_l = x_l.view(-1, self.heads, self.out_channels)
        x_r = self.get_x_r(x)
        x_r = x_r.view(-1, self.heads, self.out_channels)
        x_v = self.get_x_v(x)
        x_v = x_v.view(-1, self.heads, self.out_channels)

        num_nodes = x.size(0)
        if size_target is not None:
            num_nodes = min(num_nodes, size_target)

        if self.convolve:
            x_agg = self.get_x_agg(x_l=x_l, x_r=x_r, edge_index=edge_index, edge_weight=edge_weight)
            x_l, x_r = x_agg[:, :self.heads], x_agg[:, self.heads:]

        x_r = x_r[:num_nodes]

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index, 1.)

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        x_lr = [x_l, x_r]
        x_lr[j] = torch.cat((x_lr[j], x_v), dim=-1)

        self.convolve = False
        out = self.propagate(edge_index, x=x_lr, edge_weight=edge_weight, size=None)

        # alpha = self._alpha
        # del self._alpha
        #
        # score = self._score
        # del self._score

        out = self.update_fn(x_agg=out, x_i=x_v)

        # if return_attention_info:
        #     assert alpha is not None
        #     return out, (edge_index, alpha), score
        # else:
        return out

    def update_fn(self, x_agg, x_i):
        return self.merge_heads(x_agg) + self.bias

    def message(self, x_j: Tensor,
                x_i: Tensor, index: Tensor,
                ptr: OptTensor,
                size_i: Optional[int],
                edge_weight: OptTensor) -> Tensor:

        if self.convolve:
            return x_j

        s = x_i.size(-1)
        x_j, x_v = x_j[..., :s], x_j[..., s:]

        score = self.compute_score(x_i, x_j, index, ptr, size_i)
        self._alpha = softmax(score, index, ptr, size_i)
        self._score = score

        num_neighbors = softmax(torch.ones_like(score), index, ptr, size_i).reciprocal()

        edge_weight = 1. if edge_weight is None else edge_weight.view(-1, 1, 1)
        return x_v * (self._alpha * num_neighbors) * edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class GATv2Layer(GeneralGATLayer):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            negative_slope: float = 0.2,
            add_self_loops: bool = True,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'lcat',
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(in_channels,
                         out_channels,
                         negative_slope,
                         add_self_loops,
                         heads,
                         bias,
                         mode,
                         aggr,
                         **kwargs)

        self.lin_l = Linear(in_channels, out_channels * self.heads, bias=bias, weight_initializer='glorot')
        if share_weights_score:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, out_channels * self.heads, bias=bias, weight_initializer='glorot')

        if share_weights_value:
            self.lin_v = self.lin_l if self.flow == 'source_to_target' else self.lin_r
        else:
            self.lin_v = Linear(in_channels, out_channels * self.heads, bias=bias, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att = Parameter(torch.Tensor(1, self.heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        constant(self.att, 1.)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_v.reset_parameters()

    def get_x_r(self, x):
        return self.lin_r(x)

    def get_x_l(self, x):
        return self.lin_l(x)

    def get_x_v(self, x):
        return self.lin_v(x)

    def compute_score(self, x_i, x_j, index, ptr, size_i):
        return torch.sum(F.leaky_relu(x_i + x_j, self.negative_slope) * self.att, dim=-1, keepdim=True)


class GAT2v2Layer(GATv2Layer):
    x_0: OptTensor
    def __init__(
            self,
            channels: Union[int, Tuple[int, int]],
            alpha: float,
            theta: float = None,
            layer: int = None,
            normalize: bool = True,
            add_self_loops: bool = True,
            negative_slope: float = 0.2,
            heads: int = 1,
            bias: bool = True,
            mode: str = 'gcn',
            share_weights_score: bool = False,
            share_weights_value: bool = False,
            **kwargs,
    ):
        assert share_weights_value  # TODO

        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels=channels, out_channels=channels // heads, heads=heads, negative_slope=negative_slope,
                         add_self_loops=add_self_loops, bias=bias, mode=mode, share_weights_score=share_weights_score,
                         share_weights_value=share_weights_value, **kwargs)
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.normalize = normalize
        self.x_0 = None

    def get_x_v(self, x):
        return x

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, size_target: int = None,
                edge_weight: OptTensor = None, return_attention_info: bool = False):
        assert not return_attention_info  # TODO
        x_0 = x
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops,  dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

        self.x_0 = x_0
        return super().forward(x, edge_index, size_target, edge_weight, return_attention_info)

    def update_fn(self, x_agg, x_i):
        x_0 = self.x_0
        x_agg = self.merge_heads(x_agg)

        x_agg.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x_agg.size(0)]

        x_agg = x_agg.add_(x_0)
        x_agg = torch.addmm(x_agg, x_agg, self.lin_v.weight, beta=1. - self.beta, alpha=self.beta)

        return x_agg

