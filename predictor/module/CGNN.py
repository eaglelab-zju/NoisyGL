import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader


class CGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(CGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.mlp = torch.nn.Linear(out_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.temperature = 0.5

    def forward(self, x, edge_index):
        # First layer
        h_k_1 = x
        p_k = self.aggregate(self.conv1, h_k_1, edge_index)
        h_k = self.combine(h_k_1, p_k)

        # Apply dropout
        h_k = self.dropout(h_k)

        # Second layer
        h_k_1 = h_k
        p_k = self.aggregate(self.conv2, h_k_1, edge_index)
        h_k = self.combine(h_k_1, p_k)

        # Apply dropout
        h_k = self.dropout(h_k)

        # Third layer
        h_k_1 = h_k
        p_k = self.aggregate(self.conv3, h_k_1, edge_index)
        h_k = self.combine(h_k_1, p_k)

        return h_k

    def aggregate(self, conv_layer, h_k_1, edge_index):
        # Aggregation operation using GCNConv layer
        return conv_layer(h_k_1, edge_index)

    def combine(self, h_k_1, p_k):
        # Combination operation
        return F.relu(p_k)


def custom_dropout_features(x, p=0.5, training=True):
    if training:
        mask = torch.bernoulli(torch.full_like(x, 1 - p))
        return x * mask
    else:
        return x


def contrastive_loss(out_edge_dropout, out_node_dropout, temperature=0.5):
    loader = DataLoader(range(out_edge_dropout.shape[0]), batch_size=1000, shuffle=True)
    loss = 0
    for index in loader:
        sim_edge = F.cosine_similarity(out_edge_dropout[index].unsqueeze(1), out_edge_dropout[index].unsqueeze(0), dim=-1)
        sim_node = F.cosine_similarity(out_node_dropout[index].unsqueeze(1), out_node_dropout[index].unsqueeze(0), dim=-1)
        sim_edge_node = F.cosine_similarity(out_edge_dropout[index].unsqueeze(1), out_node_dropout[index].unsqueeze(0), dim=-1)
        part1 = torch.exp(sim_edge_node / temperature) / torch.sum(torch.exp(sim_node / temperature), dim=-1, keepdim=True)
        part2 = torch.exp(sim_edge_node / temperature) / torch.sum(torch.exp(sim_edge / temperature), dim=-1, keepdim=True)
        current_loss = -(torch.log(part1) + torch.log(part2)) / 2
        loss += current_loss.mean()
    '''
    sim_edge = F.cosine_similarity(out_edge_dropout.unsqueeze(1), out_edge_dropout.unsqueeze(0), dim=-1)
    sim_node = F.cosine_similarity(out_node_dropout.unsqueeze(1), out_node_dropout.unsqueeze(0), dim=-1)
    sim_edge_node = F.cosine_similarity(out_edge_dropout.unsqueeze(1), out_node_dropout.unsqueeze(0), dim=-1)
    part1 = torch.exp(sim_edge_node / temperature) / torch.sum(torch.exp(sim_node / temperature), dim=-1, keepdim=True)
    part2 = torch.exp(sim_edge_node / temperature) / torch.sum(torch.exp(sim_edge / temperature), dim=-1, keepdim=True)
    loss = -(torch.log(part1) + torch.log(part2)) / 2
    '''
    return loss


def supervised_loss(out, y):
    return F.cross_entropy(out, y)


def clean_noisy_labels(model, features, adj, y, train_mask,num_nodes, threshold=0.8):
    # Obtain pseudo-labels
    model.eval()
    with torch.no_grad():
        out = model(features, adj)
    # Calculate consistency scores

    # adj = to_dense_adj(adj)[0]
    # similarity = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)
    # sparse_adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones(adj.shape[1]).to(adj.device), size=(num_nodes, num_nodes)).coalesce()
    noisy_labels = y.clone()
    for i in range(num_nodes):
        if i in train_mask:
            continue
        neighbors = adj[i].coalesce().indices().squeeze()
        # neighbors = adj[i].nonzero(as_tuple=False).squeeze()
        neighbor_labels = y[neighbors]
        if neighbor_labels.numel() == 0:  # Check if neighbor_labels is an empty tensor
            continue
        most_common_label = torch.mode(neighbor_labels).values.item()
        if most_common_label != y[i].item():
            emb1 = out[i]
            if len(neighbors.shape) != 0:
                emb2 = out[neighbors]
            else:
                emb2 = out[neighbors].unsqueeze(0)
            #similarity_neighbors = similarity[i][neighbors]
            #similar_neighbors = similarity_neighbors > threshold
            similarity_neighbors = F.cosine_similarity(emb1, emb2)
            similar_neighbors = similarity_neighbors > threshold
            if similar_neighbors.float().mean().item() > threshold:
                noisy_labels[i] = most_common_label

    # # Find noisy labels
    # row, col = adj.coalesce().indices()
    # neighbor_labels = torch.zeros_like(y)
    # neighbor_labels[row] = y[col]
    #
    # # Calculate the most common neighbor label
    # unique_labels, counts = torch.unique(neighbor_labels[row], return_counts=True)
    # most_common_label = unique_labels[counts.argmax()]
    #
    # noisy_nodes = (most_common_label != y).flatten()
    # # Calculate consistency scores
    # emb1 = out.unsqueeze(1).repeat(1, col.size(0), 1)
    # emb2 = out[col, :]
    # similarity_neighbors = F.cosine_similarity(emb1, emb2.unsqueeze(0), dim=-1)
    # similar_neighbors = similarity_neighbors > threshold
    # is_consistent = similar_neighbors.float().mean(dim=1) > threshold
    # is_noisy = noisy_nodes & is_consistent
    # noisy_labels = y.clone()
    # noisy_labels[is_noisy] = neighbor_labels[is_noisy]
    # noisy_labels[train_mask] = y[train_mask]
    return noisy_labels
