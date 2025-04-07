import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
from utils.tools import setup_seed
from scipy import stats


def uniform_noise_cp(n_classes, noise_rate):
    P = np.float64(noise_rate) / np.float64(n_classes - 1) * np.ones((n_classes, n_classes))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise_rate)) * np.ones(n_classes))
    diag_idx = np.arange(n_classes)
    P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def pair_noise_cp(n_classes, noise_rate):
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        P[i, i - 1] = np.float64(noise_rate)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def random_noise_cp(n_classes, noise_rate):
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        tp = np.random.rand(n_classes)
        tp[i] = 0
        tp = (tp / tp.sum()) * noise_rate
        P[i, :] += tp
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def label_dropout(masks, dropout_rate, random_seed):
    new_masks = []
    for mask in masks:
        n_labels = len(mask)
        idx = np.arange(n_labels)
        # rs = np.random.RandomState(random_seed)
        # rs.shuffle(idx)
        idx = idx[0: int((1 - dropout_rate) * n_labels)]
        idx.sort()
        new_masks.append(mask[idx])

    return new_masks


def add_instance_independent_label_noise(labels, cp, random_seed):
    assert_array_almost_equal(cp.sum(axis=1), np.ones(cp.shape[1]))
    n_labels = labels.shape[0]
    noisy_labels = labels.copy()
    rs = np.random.RandomState(random_seed)

    for i in range(n_labels):
        label = labels[i]
        flipped = rs.multinomial(1, cp[label, :], 1)[0]
        noisy_label = np.where(flipped == 1)[0]
        noisy_labels[i] = noisy_label

    return noisy_labels


def add_instance_dependent_label_noise(noise_rate, feature, labels, num_classes, norm_std, seed):
    '''
    Add instance-dependent label noise to the labels.
    Implemented according to the following paper:
    Xia, Xiaobo, et al. "Part-dependent label noise: Towards instance-dependent label noise." Advances in Neural Information Processing Systems 33 (2020): 7597-7610.
    paper link: https://proceedings.neurips.cc/paper_files/paper/2020/hash/5607fe8879e4fd269e88387e8cb30b7e-Abstract.html
    code link: https://github.com/xiaoboxia/Part-dependent-label-noise

    Parameters
    ----------
    noise_rate: int
        The number of label classes
    feature: torch.Tensor
        Node features
    labels: np.ndarray
        Original labels
    num_classes: int
        The number of label classes
    norm_std: float
        Hyperparameter
    seed: int
        Set random seed

    Returns
    ------
    new_label: np.ndarray
        Processed noisy labels
    '''
    label_num = num_classes
    setup_seed(seed)
    num_nodes = labels.shape[0]
    feature_size = feature.shape[1]

    P = []
    flip_distribution = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate, scale=norm_std)
    flip_rate = flip_distribution.rvs(num_nodes)

    labels = torch.Tensor(labels).to(torch.long)
    labels = labels.to(feature.device)

    W = np.random.randn(label_num, feature_size, label_num)
    W = torch.FloatTensor(W).to(feature.device)

    for i in range(num_nodes):
        # 1*m *  m*10 = 1*10
        x = feature[i].unsqueeze(0)
        y = labels[i]
        A = x.mm(W[y]).squeeze(0)
        A[y] = -torch.inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    return np.array(new_label)



def label_process(labels, features, n_classes, noise_type='uniform', noise_rate=0, random_seed=5, debug=True):
    '''
    Parameters
    ----------
    labels: np.ndarray
        Original labels
    features: torch.Tensor
        Node features
    n_classes: int
        The number of label classes
    noise_type: string
        Specify the type of label noise
    noise_rate: float
        Specify label noise rate
    random_seed: int
        Set random seed
    debug: bool
        Debug mode

    Returns
    -------
    noisy_train_labels: np.ndarray
        Processed noisy labels
    modified_mask: np.ndarray
        Mark modified labels

    '''
    setup_seed(random_seed)
    assert (noise_rate >= 0.) and (noise_rate <= 1.)
    # Generate label corruption probability 'cp' according to 'noise_type' and 'noise_rate'
    if debug:
        print('----label noise information:------')
    if noise_rate > 0.0:
        if noise_type == 'clean':
            if debug:
                print("Clean data")
            cp = np.eye(n_classes)
        elif noise_type == 'uniform':
            if debug:
                print("Uniform noise")
            cp = uniform_noise_cp(n_classes, noise_rate)
        elif noise_type == 'random':
            if debug:
                print("Random noise")
            cp = random_noise_cp(n_classes, noise_rate)
        elif noise_type == 'pair':
            if debug:
                print("Pair noise")
            cp = pair_noise_cp(n_classes, noise_rate)
        elif noise_type == 'instance':
            if debug:
                print("Instance dependent noise")
            cp = None
        else:
            cp = np.eye(n_classes)
            if debug:
                print("Invalid noise type for a non-zero noise rate: " + noise_type)
    else:
        cp = np.eye(n_classes)

    if noise_rate > 0.0:
        if cp is not None:
            noisy_labels = add_instance_independent_label_noise(labels.cpu().numpy(), cp, random_seed)
            add_instance_dependent_label_noise()
        else:
            noisy_labels = add_instance_dependent_label_noise(noise_rate, features, labels.cpu().numpy(), n_classes, 0.1, random_seed)
        noisy_train_labels = torch.tensor(noisy_labels).to(torch.long).to(labels.device)
    else:
        if debug:
            print('Clean data')
        noisy_train_labels = labels.clone()

    # Calculate the actual noise rate (may differ from the expected noise rate)
    actual_noise_rate = (noisy_train_labels.cpu().numpy() != labels.cpu().numpy()).mean()
    modified_mask = np.arange(labels.shape[0])[noisy_train_labels.cpu().numpy() != labels.cpu().numpy()]
    if debug:
        print('#Actual noise rate %.2f ' % actual_noise_rate)

    return noisy_train_labels, modified_mask
