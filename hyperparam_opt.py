from nni.experiment import Experiment

if __name__ == '__main__':
    experiment = Experiment('local')
    experiment.config.trial_command = 'python single_exp.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space_file = './config/_search_space/gcn.json'
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 10
    experiment.config.trial_concurrency = 2
    experiment.run(8081, debug=True)
    input('Press enter to quit')
    experiment.stop()
