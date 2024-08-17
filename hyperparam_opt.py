from nni.experiment import Experiment
from utils.tools import load_conf, save_conf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default='cora',
                    choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho', 'dblp', 'blogcatalog', 'flickr'],
                    help='Select dataset')
parser.add_argument('--method', type=str,
                    default='gcn',
                    choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching', 'apl', 'sce', 'forward', 'backward', 'lcat',
                             'nrgnn', 'rtgnn', 'cp', 'unionnet', 'cgnn', 'crgnn', 'clnode', 'rncgln', 'pignn', 'dgnn'],
                    help="Select methods")
parser.add_argument('--noise_type', type=str,
                    default='uniform',
                    choices=['clean', 'uniform', 'pair', 'asymmetric'], help='Type of label noise')
parser.add_argument('--noise_rate', type=float,
                    default='0.3',
                    help='Label noise rate')
parser.add_argument('--device', type=str,
                    default='cuda:0',
                    help='Device')
parser.add_argument('--seed', type=int,
                    default=3000,
                    help="Random Seed")
parser.add_argument('--max_trial_number', type=int,
                    default=20,
                    help="Max trial number for hyperparameter optimization")
parser.add_argument('--trial_concurrency', type=int,
                    default=10,
                    help="How many trials running at the same time")
parser.add_argument('--tuner', type=str,
                    default='TPE',
                    help="Select HPO Tuner")
parser.add_argument('--port', type=int,
                    default=8081,
                    help="The port on which NNI manager will run")
parser.add_argument('--update_config', type=bool,
                    default=True,
                    help="Update config file with optimized parameters")
args = parser.parse_args()


if __name__ == '__main__':
    experiment = Experiment('local')
    command = 'python single_exp.py'
    for k, v in sorted(vars(args).items()):
        if k in ['data', 'method', 'noise_type', 'noise_rate', 'device', 'seed']:
            command += ' --' + k + '=' + str(v)
    experiment.config.trial_command = command
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space_file = './config/_search_space/' + args.method + '.json'
    experiment.config.tuner.name = args.tuner
    experiment.config.assessor.name = 'Curvefitting'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = args.max_trial_number
    experiment.config.trial_concurrency = args.trial_concurrency
    experiment.run(args.port, debug=True)
    result = experiment.export_data()
    max_acc = 0
    opt_params = {}
    for item in result:
        if item.value > max_acc:
            max_acc = item.value
            opt_params = item.parameter
    print("highest acc")
    print(max_acc)
    print("optimized parameters")
    print(opt_params)

    if args.update_config:
        model_conf = load_conf(None, args.method, args.data)
        for item in opt_params.keys():
            if item in ['lr', 'weight_decay']:
                model_conf.training[item] = opt_params[item]
            else:
                model_conf.model[item] = opt_params[item]
        model_conf = vars(model_conf)
        save_conf(None, args.method, args.data, model_conf)
    experiment.stop()
