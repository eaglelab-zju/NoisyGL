from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GIN
import torch


class gin_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GIN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], mlp_layers=conf.model['mlp_layers'],
                         dropout=conf.model['dropout'],
                         train_eps=conf.model['train_eps']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
