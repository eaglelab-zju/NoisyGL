import torch
import os
import numpy as np
import math


def class_conditional_betweenness_centrality(data, rely_idx):
    num_classes = data.n_classes
    CBC_matrix = []
    for iter_c in range(num_classes):
        iter_c_idx = torch.nonzero(data.noisy_label[rely_idx] == iter_c).reshape(-1, )
        iter_o_idx = torch.nonzero(data.noisy_label[rely_idx] != iter_c).reshape(-1, )
        denominator = data.Pi[iter_c_idx][:, iter_o_idx]

        i_CBC = []
        for i in range(data.n_nodes):
            numerator = data.Pi[iter_c_idx, i].reshape(-1, 1) * (data.Pi[i, iter_o_idx].reshape(1, -1))
            i_CBC.append((numerator / (denominator + 1e-16)).sum().item())
        CBC_matrix.append(np.array(i_CBC) / iter_c_idx.shape[0])

    CBC_value = torch.FloatTensor(np.array(CBC_matrix)).t()
    # normalize CBC score by class
    CBC_value = CBC_value / (CBC_value.max(0).values - CBC_value.min(0).values)
    # sum the CBC value as the final CBC score
    CBC_score = CBC_value.sum(1)
    return CBC_score


def difficulty_measurer(data, rely_idx):
    node_difficulty = class_conditional_betweenness_centrality(data, rely_idx)
    return node_difficulty


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


def Personalized_PageRank(pagerank_prob, adj):
    # ppr_file = "{}/{}_ppr.pt".format(args.data_dir, args.dataset)
    # # calculating the Personalized PageRank Matrix if not exists.
    # if os.path.exists(ppr_file):
    #     Pi = torch.load(ppr_file)
    # else:
    pr_prob = 1 - pagerank_prob
    # A = index2dense(data.edge_index.cpu(), data.num_nodes)
    A = adj.to_dense()
    A_hat = A.cuda() + torch.eye(A.size(0)).cuda()  # add self-loop
    D = torch.diag(torch.sum(A_hat, 1))
    D = D.inverse().sqrt()
    A_hat = torch.mm(torch.mm(D, A_hat), D)
    Pi = pr_prob * ((torch.eye(A.size(0)).cuda() - (1 - pr_prob) * A_hat).inverse())
    Pi = Pi.cpu()
    # torch.save(Pi, ppr_file)
    return Pi


def return_statistics(confident_idx, unconfident_idx, clean_idx, corrupt_idx, args):
    tp = np.isin(confident_idx, clean_idx).sum()
    fp = np.isin(confident_idx, corrupt_idx).sum()
    fn = np.isin(unconfident_idx, clean_idx).sum()
    tn = np.isin(unconfident_idx, corrupt_idx).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 / ((1 / precision) + (1 / recall))
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sel_samples = '{}/{}'.format(tp, tp + fp)
    if args.debug == True:
        print('Selected_clean/total: {} F_Score: {} Precision: {} Accuracy: {}'.format(sel_samples, round(f1_score, 4),
                                                                                       round(precision, 4),
                                                                                       round(accuracy, 4)))
    return f1_score, precision, accuracy, recall, specificity


def index2dense(edge_index, nnode=2708):
    indx = edge_index.numpy()
    adj = np.zeros((nnode, nnode), dtype='int8')
    adj[(indx[0], indx[1])] = 1
    new_adj = torch.from_numpy(adj).float()
    return new_adj