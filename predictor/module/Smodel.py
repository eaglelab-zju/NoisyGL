import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import utils
import numpy as np
from numpy.testing import assert_array_almost_equal


class NoiseAda(nn.Module):

    def __init__(self, n_class, noise_rate_init):
        super(NoiseAda, self).__init__()
        P = torch.FloatTensor(self.build_uniform_P(n_class, noise_rate_init))
        self.B = torch.nn.parameter.Parameter(torch.log(P))

    def forward(self, pred):
        P = F.softmax(self.B, dim=1)
        return pred @ P

    def build_uniform_P(self, size, noise):
        """ The noise matrix flips any class to any other with probability
        noise / (#class - 1).
        """
        assert (noise >= 0.) and (noise <= 1.)
        P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
        np.fill_diagonal(P, (np.float64(1) - np.float64(noise)) * np.ones(size))

        diag_idx = np.arange(size)
        P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P