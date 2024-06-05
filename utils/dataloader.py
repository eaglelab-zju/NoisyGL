import torch
import dgl
import numpy as np
from utils.functional import normalize
from utils.tools import get_npz_data, get_homophily
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor, \
    AttributedGraphDataset, TUDataset, CitationFull
from torch_geometric.utils import degree
import os
import urllib.request
from .datasplit import get_split, k_fold


class Dataset:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        The name of dataset.
    feat_norm : bool
        Whether to normalize the features.
    verbose : bool
        Whether to print statistics.
    n_splits : int
        Number of data splits.
    path : str
        Path to save dataset files.
    '''

    def __init__(self, data, feat_norm=False, adj_norm=False, verbose=True, path='./data/',
                 train_size=None, val_size=None, test_size=None,
                 train_percent=None, val_percent=None, test_percent=None,
                 train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
                 add_self_loop=True, split_type='default', from_npz=False, device='cuda:0'):
        self.name = data
        self.path = path
        self.device = torch.device(device)
        self.single_graph = True
        self.self_loop = add_self_loop
        self.split_type = split_type

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.train_examples_per_class = train_examples_per_class
        self.val_examples_per_class = val_examples_per_class
        self.test_examples_per_class = test_examples_per_class

        self.prepare_data(data, feat_norm, from_npz)
        self.feats = self.feats.to(torch.float)
        # if self.single_graph:
        self.split_data(verbose)
        # else:
        #   self.split_graphs(verbose)
        # self.homophily = get_homophily(self.labels, self.adj.to_dense(), type='edge', fill=None)
        if add_self_loop:
            self.adj = self.adj + torch.eye(self.adj.shape[0], device=self.adj.device).to_sparse()
        if adj_norm:
            self.adj = normalize(self.adj, add_loop=False)
        self.adj = self.adj.coalesce()

    def prepare_data(self, ds_name, feat_norm, from_npz):
        '''
        Function to Load various datasets.
        Homophilous datasets are loaded via pyg, while heterophilous datasets are loaded with `hetero_load`.
        The results are saved as `self.feats, self.adj, self.labels, self.train_masks, self.val_masks, self.test_masks`.
        Noth that `self.adj` is undirected and has no self loops.

        Parameters
        ----------
        ds_name : str
            The name of dataset.
        feat_norm : bool
            Whether to normalize the features.
        from_npz : bool
            Whether to load data from an existing npz file.

        '''

        if from_npz:
            adj, features, labels = get_npz_data(self.path + ds_name + '.npz', self_loop=self.self_loop)
            self.adj = adj.to(self.device).coalesce()
            self.feats = features.to(self.device)
            self.labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.n_edges = self.adj.indices().shape[1] / 2
            self.n_classes = labels.max() + 1
            if feat_norm:
                self.feats = normalize(self.feats, style='row')

        elif ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph',
                         'blogcatalog', 'flickr', 'wikics', 'cornell', 'texas', 'wisconsin', 'dblp']:
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # unnormalized
            if ds_name == 'flickr':
                self.feats = self.feats.to_dense()
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse_coo_tensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                               [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges / 2
            self.n_classes = self.data_raw.num_classes

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)

            self.adj = self.adj.to(self.device)
            # normalize features
            if feat_norm:
                self.feats = normalize(self.feats, style='row')

        elif ds_name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper',
                         'roman-empire', 'wiki-cooc', 'tolokers']:
            self.feats, self.adj, self.labels, self.splits = hetero_load(ds_name, path=self.path)

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            self.adj = self.adj.to(self.device)
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.n_edges = len(self.adj.coalesce().val) / 2
            if feat_norm:
                self.feats = normalize(self.feats, style='row')
            self.n_classes = len(self.labels.unique())
        self.adj = self.adj.coalesce()
        row = self.adj.indices()[0]
        d = degree(row, self.n_nodes)
        self.ave_degree = float(torch.mean(d))

        print("""----Data statistics------'
                #Nodes %d
                #Edges %d
                #Classes %d
                #Ave_degree %.2f""" %
              (self.n_nodes, self.n_edges, self.n_classes, self.ave_degree))

    def split_data(self, verbose=True):

        '''
        Function to conduct data splitting for various datasets.

        Parameters
        ----------
        verbose : bool
            Whether to print statistics.
        '''

        self.train_masks = None
        self.val_masks = None
        self.test_masks = None
        train_type = None
        val_type = None
        test_type = None

        if self.split_type == 'default':
            if not hasattr(self.g, 'train_mask'):
                print('Split error, split type=' + self.split_type + '. Dataset ' + self.name + ' has no default split')
                exit(0)
            train_indices = torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy()
            val_indices = torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy()
            test_indices = torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy()
            train_type = 'default'
            val_type = 'default'
            test_type = 'default'
        elif self.split_type == 'percent':
            if self.train_size is not None:
                train_size = self.train_size
                train_type = 'specified'
            elif self.train_percent is not None:
                train_size = int(self.n_nodes * self.train_percent)
                train_type = str(self.train_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Train size and train percent were not configured')
                exit(0)

            if self.val_size is not None:
                val_size = self.val_size
                val_type = 'specified'
            elif self.val_percent is not None:
                val_size = int(self.n_nodes * self.val_percent)
                val_type = str(self.val_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Val size and Val percent were not configured')
                exit(0)

            if self.test_size is not None:
                test_size = self.test_size
                test_type = 'specified'
            elif self.test_percent is not None:
                test_size = int(self.n_nodes * self.test_percent)
                test_type = str(self.test_percent * 100) + ' % of nodes'
            else:
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size, )
        elif self.split_type == 'samples_per_class':
            train_size = None
            val_size = None
            test_size = None
            if self.train_examples_per_class is not None:
                train_examples_per_class = self.train_examples_per_class
                train_type = str(self.train_examples_per_class) + ' nodes per class'
            elif self.train_size is not None:
                train_examples_per_class = None
                train_size = self.train_size
                train_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Train size and train percent were not configured')
                exit(0)

            if self.val_examples_per_class is not None:
                val_examples_per_class = self.val_examples_per_class
                val_type = str(self.val_examples_per_class) + ' nodes per class'
            elif self.val_size is not None:
                val_examples_per_class = None
                val_size = self.val_size
                val_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Val size and val percent were not configured')
                exit(0)

            if self.test_examples_per_class is not None:
                test_examples_per_class = self.test_examples_per_class
                test_type = str(self.test_examples_per_class) + ' nodes per class'
            elif self.test_size is not None:
                test_examples_per_class = None
                test_size = self.test_size
                test_type = 'specified'
            else:
                test_examples_per_class = None
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_examples_per_class=train_examples_per_class,
                                                                 val_examples_per_class=val_examples_per_class,
                                                                 test_examples_per_class=test_examples_per_class,
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size)
        else:
            print('Split error: split type ' + self.split_type + ' not implemented')
            exit(0)

        self.train_masks = train_indices
        self.val_masks = val_indices
        self.test_masks = test_indices

        if verbose:
            print("""----Split statistics------'
                #Train samples %d (%s)
                #Val samples %d (%s)
                #Test samples %d (%s)""" %
                  (len(self.train_masks), train_type,
                   len(self.val_masks), val_type,
                   len(self.test_masks), test_type))

        '''
                if self.name in ['blogcatalog', 'flickr']:
            def load_obj(file_name):
                with open(file_name, 'rb') as f:
                    return pickle.load(f)

            def download(name):
                url = 'https://github.com/zhao-tong/GAug/raw/master/data/graphs/'
                try:
                    print('Downloading', url + name)
                    urllib.request.urlretrieve(url + name, os.path.join(self.path, name))
                    print('Done!')
                except:
                    raise Exception(
                        'Download failed! Make sure you have stable Internet connection and enter the right name')

            split_file = self.name + '_tvt_nids.pkl'
            if not os.path.exists(os.path.join(self.path, split_file)):
                download(split_file)
            train_indices, val_indices, test_indices = load_obj(os.path.join(self.path, split_file))
            for i in range(n_splits):
                self.train_masks.append(train_indices)
                self.val_masks.append(val_indices)
                self.test_masks.append(test_indices)

        elif self.name in ['coauthorcs', 'coauthorph', 'amazoncom', 'amazonpho']:
            for i in range(n_splits):
                np.random.seed(i)
                train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                     train_examples_per_class=20,
                                                                     val_examples_per_class=30)
                self.train_masks.append(train_indices)
                self.val_masks.append(val_indices)
                self.test_masks.append(test_indices)
        elif self.name in ['cora', 'citeseer', 'pubmed']:
            for i in range(n_splits):
                self.train_masks.append(torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy())
                self.val_masks.append(torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy())
                self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())

        elif self.name in ['amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered', 'minesweeper',
                           'roman-empire', 'wiki-cooc', 'tolokers']:
            assert n_splits < 10, 'n_splits > splits provided'
            self.train_masks = self.splits[0][:n_splits]
            self.val_masks = self.splits[1][:n_splits]
            self.test_masks = self.splits[2][:n_splits]
        elif self.name in ['ogbn-arxiv']:
            split_idx = self.data_raw.get_idx_split()
            train_idx = split_idx['train']
            val_idx = split_idx['valid']
            test_idx = split_idx['test']
            for i in range(n_splits):
                self.train_masks.append(train_idx.numpy())
                self.val_masks.append(val_idx.numpy())
                self.test_masks.append(test_idx.numpy())
        elif self.name in ['wikics']:
            for i in range(n_splits):
                self.train_masks.append(torch.nonzero(self.g.train_mask[:, i], as_tuple=False).squeeze().numpy())
                self.val_masks.append(torch.nonzero(self.g.val_mask[:, i], as_tuple=False).squeeze().numpy())
                self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())
        elif 'csbm' in self.name:
            for i in range(n_splits):
                np.random.seed(i)
                train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                     train_size=int(self.n_nodes * self.train_percent),
                                                                     val_size=int(self.n_nodes * self.val_percent))
                self.train_masks.append(train_indices)
                self.val_masks.append(val_indices)
                self.test_masks.append(test_indices)
        elif self.name in ['regression']:
            for i in range(n_splits):
                self.train_masks.append(self.masks[str(i)]['train'])
                self.val_masks.append(self.masks[str(i)]['val'])
                self.test_masks.append(self.masks[str(i)]['test'])
        elif self.name in ['cornell', 'texas', 'wisconsin']:
            for i in range(n_splits):
                self.train_masks.append(torch.nonzero(self.g.train_mask[:, i], as_tuple=False).squeeze().numpy())
                self.val_masks.append(torch.nonzero(self.g.val_mask[:, i], as_tuple=False).squeeze().numpy())
                self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())
        elif self.name in ['dblp']:
            for i in range(n_splits):
                np.random.seed(i)
                train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                     train_size=int(self.n_nodes * self.train_percent),
                                                                     val_size=int(self.n_nodes * self.val_percent))
                self.train_masks.append(train_indices)
                self.val_masks.append(val_indices)
                self.test_masks.append(test_indices)
        else:
            print('dataset not implemented')
            exit(0)
        
        '''

    def split_graphs(self, n_splits, verbose=True):
        '''
        Function to conduct data splitting for graph-level datasets.

        Parameters
        ----------
        n_splits : int
            Number of data splits.
        verbose : bool
            Whether to print statistics.

        '''
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(self.data_raw, n_splits))):
            self.train_masks.append(train_idx)
            self.val_masks.append(val_idx)
            self.test_masks.append(test_idx)


def pyg_load_dataset(name, path='./data/'):
    dic = {'cora': 'Cora',
           'citeseer': 'CiteSeer',
           'pubmed': 'PubMed',
           'amazoncom': 'Computers',
           'amazonpho': 'Photo',
           'coauthorcs': 'CS',
           'coauthorph': 'Physics',
           'wikics': 'WikiCS',
           'chameleon': 'Chameleon',
           'squirrel': 'Squirrel',
           'cornell': 'Cornell',
           'texas': 'Texas',
           'wisconsin': 'Wisconsin',
           'actor': 'Actor',
           'blogcatalog': 'blogcatalog',
           'flickr': 'flickr'}
    if name in dic.keys():
        name = dic[name]
    else:
        name = name

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=path, name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=path, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=path, name=name)
    elif name in ['WikiCS']:
        dataset = WikiCS(root=os.path.join(path, name))
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=path, name=name)
    elif name == 'Actor':
        dataset = Actor(root=os.path.join(path, name))
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=path, name=name)
    elif name in ['dblp']:
        dataset = CitationFull(root=path, name=name)
    else:
        dataset = TUDataset(root=path, name=name)
    return dataset


def hetero_load(name, path='./data/hetero_data'):
    file_name = f'{name.replace("-", "_")}.npz'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, file_name)):
        download(file_name, path)
    data = np.load(os.path.join(path, f'{name.replace("-", "_")}.npz'))
    node_features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges'])
    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    train_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in train_masks]
    val_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in val_masks]
    test_indices = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in test_masks]

    n_nodes = node_features.shape[0]
    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.long)
    graph = dgl.to_bidirected(graph)
    adj = graph.adj()

    num_classes = len(labels.unique())
    num_targets = 1 if num_classes == 2 else num_classes
    if num_targets == 1:
        labels = labels.float()

    return node_features, adj, labels, (train_indices, val_indices, test_indices)


def download(name, path):
    url = 'https://github.com/OpenGSL/HeterophilousDatasets/raw/main/data/'
    try:
        print('Downloading', url + name)
        urllib.request.urlretrieve(url + name, os.path.join(path, name))
        print('Done!')
    except:
        raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')
