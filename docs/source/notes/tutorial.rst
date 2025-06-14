:github_url: https://github.com/eaglelab-zju/NoisyGL


Build Your own model using Our Template
========================


We have provided a framework in *predictor.Base_predictor* for implementing your own GLN method.
Following shows you how to implement a vanilla GCN using our framework.

Step 1: Provide config file
----------------------------------

Before implementing your own model, you need to provide a configuration file that defines the model structure, training parameters, dataset properties, and other options.

This configuration file should be in YAML format and located in the `config` directory. You should provide configuration file for each method and dataset combination.

For example, if you want to implement a GCN model and test it on the Cora dataset, you can create a file named `mygln_cora.yaml` in the `config/mygln` directory with the following content:

.. code-block:: yaml

    model:
      method: mygln
      n_hidden: 64
      n_layer: 2
      act: F.relu
      dropout: 0.5
      norm_info: ~
      input_layer: false
      output_layer: false

    training:
      lr: 1e-2
      weight_decay: 5e-4
      n_epochs: 200
      patience: ~
      criterion: metric

    dataset:
      sparse: true

    analysis:
      flag: false
      project: NoisyGL
      save_graph: false

Step 2: Define Your Predictor Class
----------------------------------

After providing the configuration file, you can implement your own predictor class by inheriting from `Base_Predictor`.

.. code-block:: python

    from predictor.Base_Predictor import Predictor
    from predictor.module.GNNs import GCN
    import torch
    class mygln_Predictor(Predictor):
        def __init__(self, conf, data, device='cuda:0'):
            super().__init__(conf, data, device)

        def method_init(self, conf, data):
            self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                             n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                             norm_info=conf.model['norm_info'],
                             act=conf.model['act'], input_layer=conf.model['input_layer'],
                             output_layer=conf.model['output_layer']).to(self.device)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                          weight_decay=self.conf.training['weight_decay'])


Step 3: Register Your Predictor Class in *total_exp.py* and *single_exp.py*
----------------------------------

Before running your model, you need to register your predictor class in the `total_exp.py` and `single_exp.py` scripts, like this:

.. code-block:: python

    from MyGLN_predictor import mygln_Predictor
    parser.add_argument('--method', type=str,
                    default='mygln',
                    choices=['mygln'],
                    help="Select methods")


Step 4: Run Your Model
----------------------------------

After doing the above steps, you can run your model using the `single_exp.py` script for debugging.

.. code-block:: bash

    python single_exp.py --method mygln --dataset cora --noise_type uniform --noise_rate 0.1 --device cuda:0 --seed 3000

Or, if you want to run multiple datasets and noise types, you can use the `total_exp.py` script:

.. code-block:: bash

    python total_exp.py --methods mygln --datasets cora citeseer --noise_type uniform --noise_rate 0.1 0.2 --device cuda:0 --seed 3000

This will execute your model on the Cora dataset with a uniform noise type and a noise rate of 0.1, using the specified device and seed.
You can also customize the command line arguments to suit your needs, such as changing the dataset, noise type, or device.

Step 5: Analyze Results
----------------------------------
After running your model, you can analyze the results in the `log` directory.

