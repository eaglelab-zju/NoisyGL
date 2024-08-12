from nni.experiment import Experiment
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default='cora',
                    choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho', 'dblp', 'blogcatalog', 'flickr'],
                    help='Select dataset')
parser.add_argument('--method', type=str,
                    default='gcn',
                    choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching', 'apl', 'sce', 'forward', 'backward',
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
args = parser.parse_args()


if __name__ == '__main__':
    experiment = Experiment('local')
    command = 'python single_exp.py'
    for k,v in sorted(vars(args).items()):
        command += ' --' + k + '=' + str(v)
    experiment.config.trial_command = command
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space_file = './config/_search_space/gcn.json'
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 4
    experiment.config.trial_concurrency = 2
    experiment.run(8081, debug=True)
    input('Press enter to quit')
    experiment.stop()
