import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv


class CRGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CRGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class ProjectionHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectionHead, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassificationHead, self).__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=-1)


def custom_dropout_features(x, p=0.5, training=True):
    if training:
        mask = torch.bernoulli(torch.full_like(x, 1 - p))
        return x * mask
    else:
        return x


def contrastive_loss(z1, z2, tau):
    # Compute contrastive loss
    # z1, z2: embeddings (torch.Tensor)
    # tau: temperature parameter
    N = z1.size(0)
    pos1 = torch.exp(F.cosine_similarity(z1, z2, dim=1) / tau)
    neg1 = torch.sum(torch.exp(torch.mm(z1, z1.t()) / tau), dim=1) + pos1
    pos2 = torch.exp(F.cosine_similarity(z2, z1, dim=1) / tau)
    neg2 = torch.sum(torch.exp(torch.mm(z2, z2.t()) / tau), dim=1) + pos2
    loss = (torch.sum(-torch.log(pos1 / (pos1 + neg1))) + torch.sum(-torch.log(pos2 / (pos2 + neg2)))) / (2 * N)
    return loss


def dynamic_cross_entropy_loss(p1, p2, labels):
    labels = labels.long()
    pseudo_labels1 = p1.argmax(dim=1)
    pseudo_labels2 = p2.argmax(dim=1)
    consistent_mask = (pseudo_labels1 == pseudo_labels2)
    loss = F.cross_entropy(p1[consistent_mask], labels[consistent_mask])
    return loss


def cross_space_consistency_loss(zm, pm):
    # Compute cross-space consistency loss
    # zm, pm: similarity matrices (torch.Tensor)
    loss = F.mse_loss(zm, pm)
    return loss