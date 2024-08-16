import argparse
import nni
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed, get_neighbors
from utils.labelnoise import label_process
from predictor.NRGNN_Predictor import nrgnn_Predictor
from predictor.CP_Predictor import cp_Predictor
from predictor.Smodel_Predictor import smodel_Predictor
from predictor.Coteaching_Predictor import coteaching_Predictor
from predictor.GCN_Predictor import gcn_Predictor
from predictor.RTGNN_Predictor import rtgnn_Predictor
from predictor.CLNode_Predictor import clnode_Predictor
from predictor.RNCGLN_Predictor import rncgln_Predictor
from predictor.PIGNN_Predictor import pignn_Predictor
from predictor.GIN_Predictor import gin_Predictor
from predictor.DGNN_Predictor import dgnn_Predictor
from predictor.UnionNET_Predictor import unionnet_Predictor
from predictor.CGNN_Predictor import cgnn_Predictor
from predictor.JoCoR_Predictor import jocor_Predictor
from predictor.CRGNN_Predictor import crgnn_Predictor
from predictor.APL_Predictor import apl_Predictor
from predictor.SCE_Predictor import sce_Predictor
from predictor.Forward_Predictor import forward_Predictor
from predictor.Backward_Predictor import backward_Predictor
from predictor.LCAT_Predictor import lcat_Predictor


def merge_params(model_conf):
    turner_params = nni.get_next_parameter()
    print(turner_params)
    for item in turner_params.keys():
        print(item)
        if item in ['lr', 'weight_decay']:
            model_conf.training[item] = turner_params[item]
        else:
            model_conf.model[item] = turner_params[item]
    print(model_conf)
    return model_conf


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default='cora',
                    choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                             'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                    help='Select dataset')
parser.add_argument('--method', type=str,
                    default='gcn',
                    choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching', 'apl', 'sce', 'forward', 'backward', 'lcat',
                             'nrgnn', 'rtgnn', 'cp', 'unionnet', 'cgnn', 'crgnn', 'clnode', 'rncgln', 'pignn', 'dgnn'],
                    help="Select methods")
parser.add_argument('--noise_type', type=str,
                    default='uniform',
                    choices=['clean', 'uniform', 'pair', 'random'], help='Type of label noise')
parser.add_argument('--noise_rate', type=float,
                    default='0.3',
                    help='Label noise rate')
parser.add_argument('--device', type=str,
                    default='cuda:0',
                    help='Device')
parser.add_argument('--seed', type=int,
                    default=3000,
                    help="Random Seed")
args = parser.parse_args()



if __name__ == '__main__':
    print(args)
    data_path = './data/'
    data_conf = load_conf('./config/_dataset/' + args.data + '.yaml')
    if nni.get_trial_id() == "STANDALONE":
        setup_seed(args.seed)
    data = Dataset(args.data, path=data_path,
                   feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                   train_size=data_conf.split['train_size'],
                   val_size=data_conf.split['val_size'],
                   test_size=data_conf.split['test_size'],
                   train_percent=data_conf.split['train_percent'],
                   val_percent=data_conf.split['val_percent'],
                   test_percent=data_conf.split['test_percent'],
                   train_examples_per_class=data_conf.split['train_examples_per_class'],
                   val_examples_per_class=data_conf.split['val_examples_per_class'],
                   test_examples_per_class=data_conf.split['test_examples_per_class'],
                   add_self_loop=data_conf.modify['add_self_loop'],
                   from_npz=data_conf.modify['from_npz_largest_component'],
                   device=args.device,
                   split_type=data_conf.split['split_type'])
    print('Current device: ' + str(data.feats.device))
    model_conf = load_conf(None, args.method, data.name)
    if nni.get_trial_id() != "STANDALONE":
        model_conf = merge_params(model_conf)
    data.noisy_label, modified_mask = label_process(labels=data.labels, n_classes=data.n_classes,
                                                    noise_type=args.noise_type, noise_rate=args.noise_rate,
                                                    random_seed=args.seed, debug=True)
    model_conf.model['n_feat'] = data.dim_feats
    model_conf.model['n_classes'] = data.n_classes
    model_conf.training['debug'] = True
    predictor = eval(args.method + '_Predictor')(model_conf, data, args.device)
    result = predictor.train()
    if nni.get_trial_id() != "STANDALONE":
        nni.report_final_result(float(result['test']))
