import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from predictor.module.GNNs import GCN, MLP
import math


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5, norm_info=None,
                 act='F.relu', input_layer=False, output_layer=False, bias=True):
        super(GCNNet, self).__init__()
        self.GCN = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                       n_layers=n_layers, dropout=dropout, norm_info=norm_info,
                       act=act, input_layer=input_layer,
                       output_layer=output_layer, bias=bias)

    def forward(self, x, adj):
        x = self.GCN(x, adj)
        return F.log_softmax(x, dim=1)


class GCNClassifier(torch.nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_embedding, dropout):
        super().__init__()
        self.gnn = GCNNet(in_channels=n_feat, hidden_channels=n_hidden, out_channels=n_embedding, dropout=dropout)
        self.mlp = MLP(in_channels=n_embedding, hidden_channels=n_hidden, out_channels=n_class, dropout=dropout, n_layers=1)

    def forward(self, x, adj):
        embedding = self.gnn(x, adj)
        out = self.mlp(embedding)
        return embedding, out


class DifficultyMeasurer:
    def __init__(self, device, alpha=1):
        self.device = device
        self.alpha = alpha

    def measure_difficulty(self, edge_index, train_id, label, embedding):
        local_difficulty = self.neighborhood_difficulty_measurer(edge_index, train_id, label)
        global_difficulty = self.feature_difficulty_measurer(train_id, label, embedding)
        node_difficulty = local_difficulty + self.alpha * global_difficulty
        return node_difficulty

    def neighborhood_difficulty_measurer(self, edge_index, train_id, label):
        # 加上自环，将节点本身的标签也计算在内
        neighbor_label, _ = add_self_loops(edge_index)
        # 得到每个节点的邻居标签
        neighbor_label[1] = label[neighbor_label[1]]
        # 得到训练集中每个节点的邻节点分布
        neighbor_label = torch.transpose(neighbor_label, 0, 1)
        index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)
        neighbor_class = torch.sparse_coo_tensor(index.T, count)
        neighbor_class = neighbor_class.to_dense().float()
        # 开始计算节点的邻居信息熵
        neighbor_class = neighbor_class[train_id]
        neighbor_class = F.normalize(neighbor_class, 1.0, 1)
        neighbor_entropy = -1 * neighbor_class * torch.log(
            neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
        local_difficulty = neighbor_entropy.sum(1)
        return local_difficulty.to(self.device)

    # feature-based difficulty measurer
    def feature_difficulty_measurer(self, train_id, label, embedding):
        normalized_embedding = F.normalize(torch.exp(embedding))
        classes = label.unique()
        class_features = {}
        for c in classes:
            class_nodes = torch.nonzero(label == c).squeeze(1)
            node_features = normalized_embedding.index_select(0, class_nodes)
            class_feature = node_features.sum(dim=0)
            # 这里注意归一化
            class_feature = class_feature / torch.sqrt((class_feature * class_feature).sum())
            class_features[c.item()] = class_feature

        similarity = {}
        for u in train_id:
            # 做了实验，认为让节点乘错误的类别feature，看看效果
            feature = normalized_embedding[u]
            class_feature = class_features[label[u].item()]
            sim = torch.dot(feature, class_feature)
            sum = torch.tensor(0.).to(self.device)
            for cf in class_features.values():
                sum += torch.dot(feature, cf)
            sim = sim * len(classes) / sum
            similarity[u.item()] = sim

        class_avg = {}
        for c in classes:
            count = 0.
            sum = 0.
            for u in train_id:
                if label[u] == c:
                    count += 1
                    sum += similarity[u.item()]
            class_avg[c.item()] = sum / count

        global_difficulty = []

        for u in train_id:
            sim = similarity[u.item()] / class_avg[label[u].item()]
            # print(u,sim)
            sim = torch.tensor(1) if sim > 0.95 else sim
            node_difficulty = 1 / sim
            global_difficulty.append(node_difficulty)

        return torch.tensor(global_difficulty).to(self.device)

    def sort_training_nodes(self, edge_index, train_id, label, embedding):
        node_difficulty = self.measure_difficulty(edge_index, train_id, label, embedding)
        _, sorted_trainset_indices = torch.sort(node_difficulty)
        sorted_trainset = train_id[sorted_trainset_indices.cpu().numpy()]
        return sorted_trainset, sorted_trainset_indices


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
