import torch
import numpy as np
import random
import argparse
import os
import ruamel.yaml as yaml
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


def load_conf(path: str = None, method: str = None, dataset: str = None):
    '''
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    '''
    if path == None and method == None:
        raise KeyError
    if path == None and dataset == None:
        raise KeyError
    if path == None:
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        path = os.path.join(dir, method, method + '_' + dataset + ".yaml")
        if os.path.exists(path) == False:
            raise KeyError("The method configuration file is not provided.")

    conf = open(path, "r").read()
    conf = yaml.load(conf)
    conf = argparse.Namespace(**conf)
    return conf


def save_conf(path: str = None, method: str = None, dataset: str = None, conf: any = None):
    '''
    Function to load config file.

    Parameters
    ----------
    path : str
        Path to load config file. Load default configuration if set to `None`.
    method : str
        Name of the used mathod. Necessary if ``path`` is set to `None`.
    dataset : str
        Name of the corresponding dataset. Necessary if ``path`` is set to `None`.
    conf : argparse.Namespace
        The config file to save.

    Returns
    -------
        None
    '''

    if path == None and method == None:
        raise KeyError
    if path == None and dataset == None:
        raise KeyError
    if path == None:
        dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        path = os.path.join(dir, method, method + '_' + dataset + ".yaml")

    with open(path, 'w') as f:
        yaml.safe_dump(conf, f, default_flow_style=False)
    print('config file ' + path + ' updated')
    return None

def get_npz_data(file_name, self_loop):
    '''
    Load data from a .npz file and process it to obtain adjacency matrix, features, and labels.

    Parameters
    ----------
    file_name: str
        The path to the .npz file containing the data.
    self_loop: bool
        whether to add self-loops to the adjacency matrix.

    Returns
    -------
    adj: torch.sparse.FloatTensor
        The adjacency matrix in sparse format.
    features: torch.FloatTensor
        The node features in dense format.
    labels: np.ndarray
        The labels for the nodes.
    '''

    adj, features, labels = load_npz(file_name)
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1
    lcc = largest_connected_components(adj)
    adj = adj[lcc][:, lcc]
    if not self_loop:
        adj.setdiag(0)
    else:
        adj.setdiag(1)
    features = features[lcc]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = labels[lcc]
    return adj, features, labels


def load_npz(file_name, is_sparse=True):
    '''
    Load data from a .npz file.

    Parameters
    ----------
    file_name: str
        The path to the .npz file containing the data.
    is_sparse: bool
        Whether the adjacency matrix is sparse or dense. If True, the adjacency matrix will be loaded as a sparse matrix.

    Returns
    -------
    adj: scipy.sparse.csr_matrix or np.ndarray
        The adjacency matrix of the graph. If `is_sparse` is True, it will be a sparse matrix; otherwise, it will be a dense matrix.
    features: scipy.sparse.csr_matrix or np.ndarray
        The node features of the graph. If the features are not present in the file, it defaults to None.
    labels: np.ndarray or None
        The labels of the nodes. If labels are not present in the file, it defaults to None.
    '''

    with np.load(file_name) as loader:
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels


def largest_connected_components(adj, n_components=1):
    """
    Select k largest connected components.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        input adjacency matrix
    n_components : int
        n largest connected components we want to select
    Returns
    -------
    nodes_to_keep : list
        list of indices of nodes to keep in the largest connected components

    """

    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    Parameters
    ----------
    sparse_mx : scipy.sparse.csr_matrix
        The input sparse matrix in CSR format.
    Returns
    -------
    torch.sparse.FloatTensor
        The converted torch sparse tensor.

    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_node_homophily(label, adj):
    '''
    Calculate the node homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    homophily : torch.float
        The node homophily of the graph.

    '''
    label = label.cpu().numpy()
    adj = adj.cpu().numpy()
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = (np.multiply((label == label.T), adj)).sum(axis=1)
    d = adj.sum(axis=1)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1. / d[i])
    return np.mean(homos)


def get_edge_homophily(label, adj):
    '''
    Calculate the node homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.

    Returns
    -------
    homophily : torch.float
        The edge homophily of the graph.

    '''
    num_edge = adj.sum()
    cnt = 0
    for i, j in adj.nonzero():
        if label[i] == label[j]:
            cnt += adj[i, j]
    return cnt/num_edge


def get_homophily(label, adj, type='node', fill=None):
    '''
    Calculate node or edge homophily of a graph.

    Parameters
    ----------
    label : torch.tensor
        The ground truth labels.
    adj : torch.tensor
        The adjacency matrix in dense form.
    type : str
        This decides whether to calculate node homo or edge homo.
    fill : str
        The value to fill in the diagonal of `adj`. If set to `None`, the operation won't be done.

    Returns
    -------
    homophily : np.float
        The node or edge homophily of a graph.

    '''
    if fill:
        np.fill_diagonal(adj, fill)
    return eval('get_'+type+'_homophily(label, adj)')


def setup_seed(seed):
    '''
    Setup random seed so that the experimental results are reproducible

    Parameters
    ----------
    seed : int
        random seed for torch, numpy and random

    Returns
    -------
    None
    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_neighbors(adj, mask):
    '''
    This function extracts the unique neighbors of nodes specified by a mask from a sparse adjacency matrix.

    Parameters
    ----------
    adj : torch.sparse.FloatTensor
        The adjacency matrix in sparse format.
    mask : np.ndarray
        The mask of nodes to find neighbors for.

    Returns
    -------
    neighbors : np.ndarray
        The unique neighbors of the nodes specified by the mask.
    '''

    edge_index = adj.indices()
    row, col = edge_index[0, :].cpu().numpy(), edge_index[1, :].cpu().numpy()
    col_mask = np.in1d(row, mask)
    masked_col = col[col_mask]
    neighbors = np.unique(masked_col)
    return neighbors


def heatmap(matrix, title, n_classes):
    '''
    Draw a heatmap of the given matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to visualize.
    title: str
        The title of the heatmap.
    n_classes: int
        The number of classes in the matrix.

    Returns
    -------
    None
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    '''
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="w", fontsize=5)
    '''

    ax.set_title(title, fontdict={'weight': 'normal', 'size': 25})
    fig.tight_layout()
    plt.show()
    fig.savefig('./data_eval/' + title + '.png', format='png')