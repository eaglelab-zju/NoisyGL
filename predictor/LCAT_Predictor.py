from predictor.Base_Predictor import Predictor
from predictor.module.LCAT import GATv2Layer, GAT2v2Layer, LCAT
import torch
import torch.nn.functional as F
import time
import nni
from copy import deepcopy


class lcat_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = LCAT(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                          out_channels=conf.model['n_classes'], heads=conf.model['heads'],
                          alpha=conf.model['alpha'], theta=conf.model["theta"],
                          negative_slope=conf.model['negative_slope'], module=conf.model['module'],
                          dropout=conf.model["dropout"]).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])


