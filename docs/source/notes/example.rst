:github_url: https://github.com/eaglelab-zju/NoisyGL


Quick Start Guide
========================

There are three basic scripts in NoisyGL, namely

* single_exp.py

* total_exp.py

* hyperparameter_opt.py

Following examples show you how to use these tools.

Run multiple Experiments and record all results in Benchmark Mode
--------------------------------------------------------------------

.. code-block:: bash

    python total_exp.py --runs 10 --methods gcn gin --datasets cora citeseer pubmed --noise_type clean uniform pair --noise_rate 0.1 0.2 --device cuda:0 --seed 3000

By running the command above, two methods 'gcn' and 'gin' will be tested on three datasets 'cora', 'citeseer', and 'pubmed' under different types and rates of label noise.
Each experiment will run 10 times and the total results will be saved at ./log and named by the current timestamp.
You can customize the combination of method, data, noise type, and noise rate by changing the corresponding arguments.

Run a Single Experiment in Debug Mode
-------------------------------------------------------------------------------

.. code-block:: bash

    python single_exp.py --method gcn --data cora --noise_type uniform --noise_rate 0.1 --device cuda:0 --seed 3000

This command runs a single experiment in debug mode and is usually used for debugging.
By running this, detailed experiment information will be printed on the terminal, which can be used to locate the problem.

When design your own customized predictor, or analyze the intermediate state of the model, you can create code blocks that only execute in debug mode in the following way:

.. code-block:: python

    if self.conf.training['debug']:
        print("break point")

Run Hyperparameter Optimization
-------------------------------------------------------------------------------

.. code-block:: bash

    python hyperparam_opt.py --method gcn --data cora --noise_type uniform --noise_rate 0.1 --device cuda:0 --max_trial_number 20 --trial_concurrency 4 --port 8081 --update_config True

By running the command above, an NNI manager will run on http://localhost:8081,
then automatically run 20 HPO trails, each trail call 'single_exp.py' with different hyperparameters.
After all HPO trials are finished,
a new config file with optimized hyperparameters will overwrite the original one at "./config/gcn/gcn_cora.yaml".
You can optimize hyperparameters for different methods on various datasets and noise types
by changing the corresponding arguments.

