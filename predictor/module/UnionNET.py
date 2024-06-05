import torch
import torch.nn.functional as F


def construct_support_set(features, train_mask, y, edge_index, k):
    support_set = torch.zeros(features.size(0), k, features.size(1)).to(features.device)
    labels = torch.zeros(features.size(0), k, 1).to(y.device)
    for i, is_labeled in enumerate(train_mask):
        if is_labeled:
            neighbors = edge_index[1][edge_index[0] == i]  # Get neighbors of node i
            if len(neighbors) >= k:
                anchor_node = features[i].unsqueeze(0)  # shape: (1, embedding_dim)
                # Calculate similarities between the current node and its neighbors
                similarities = torch.mm(features[neighbors], anchor_node.T).squeeze()
                # Select the indices of the k-nearest neighbors
                _, topk_indices = torch.topk(similarities, k=k, largest=True)
                # Populate the support set with the features of the k-nearest neighbors
                support_set[i] = features[neighbors][topk_indices]
                labels[i] = y[neighbors[topk_indices]].view(k, 1)
    return support_set, labels


def label_aggregation(S, y, h_x_hat):
    class_prob = torch.zeros(y.size(0), len(y[0][0][0])).to(h_x_hat.device)
    for i in range(y.size(0)):
        prob = 0
        inner_sum = sum([torch.exp(torch.dot(h_x_j.view(-1), h_x_hat[i].T.view(-1))) for h_x_j in S])
        for j in S[i]:
            prob += torch.exp(torch.dot(j, h_x_hat[i].T.view(-1))) * y[i]
        class_prob[i] = (prob[0][0] / inner_sum)
    return class_prob


def label_correction_loss(S, y, h_x_hat, class_prob):
    total_loss = 0
    for i, h_x_i in enumerate(h_x_hat):
        inner_sum = torch.sum(torch.exp(torch.matmul(S, h_x_i)), dim=0)
        p_c = torch.max(torch.exp(torch.matmul(S, h_x_i)) / inner_sum)
        y_c = torch.argmax(p_c).item()
        y_c_one_hot = F.one_hot(torch.tensor(y_c), num_classes=len(y.unique())).to(y.device)
        total_loss += y[i] * y_c_one_hot * class_prob[i]
    return -total_loss.sum()