import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

def accuracy(labels, logits):
    '''
    Compute the accuracy score given true labels and predicted labels.

    Parameters
    ----------
    labels: np.array
        Ground truth labels.
    logits : np.array
        Predicted labels.

    Returns
    -------
    accuracy : np.float
        The Accuracy score.

    '''
    return np.sum(logits.argmax(1)==labels)/len(labels)

def normalize(mx, style='symmetric', add_loop=True, p=None):
    '''
    Normalize the feature matrix or adj matrix.

    Parameters
    ----------
    mx : torch.tensor
        Feature matrix or adj matrix to normalize. Note that either sparse or dense form is supported.
    style: str
        If set as ``row``, `mx` will be row-wise normalized.
        If set as ``symmetric``, `mx` will be normalized as in GCN.
        If set as ``softmax``, `mx` will be normalized using softmax.
        If set as ``row-norm``, `mx` will be normalized using `F.normalize` in pytorch.
    add_loop : bool
        Whether to add self loop.
    p : float
        The exponent value in the norm formulation. Onlu used when style is set as ``row-norm``.
    Returns
    -------
    normalized_mx : torch.tensor
        The normalized matrix.
    '''
    if style == 'row':
        if mx.is_sparse:
            return row_normalize_sp(mx)
        else:
            return row_nomalize(mx)
    elif style == 'symmetric':
        if mx.is_sparse:
            return normalize_sp_tensor_tractable(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)
    elif style == 'softmax':
        if mx.is_sparse:
            return torch.sparse.softmax(mx, dim=-1)
        else:
            return F.softmax(mx, dim=-1)
    elif style == 'row-norm':
        assert p is not None
        if mx.is_sparse:
            # TODO
            pass
        else:
            return F.normalize(mx, dim=-1, p=p)
    else:
        raise KeyError("The normalize style is not provided.")


def row_nomalize(mx):
    """Row-normalize sparse matrix.
    """
    # device = mx.device
    # mx = mx.cpu().numpy()
    # r_sum = np.array(mx.sum(1))
    # r_inv = np.power(r_sum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)
    # mx = torch.tensor(mx).to(device)

    r_sum = mx.sum(1)
    r_inv = r_sum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx

    return mx


def row_normalize_sp(mx):
    adj = mx.coalesce()
    inv_sqrt_degree = 1. / (torch.sparse.sum(mx, dim=1).values() + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def normalize_sp_tensor_tractable(adj, add_loop=True):
    n = adj.shape[0]
    device = adj.device
    if add_loop:
        adj = adj + torch.eye(n, device=device).to_sparse()
    adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value
    return torch.sparse_coo_tensor(adj.indices(), new_values, adj.size())


def normalize_tensor(adj, add_loop=True):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0]).to(device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A



def normalize_sp_matrix(adj, add_loop=True):
    mx = adj + sp.eye(adj.shape[0]) if add_loop else adj
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    new = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return new
