from predictor.Base_Predictor import Predictor
from predictor.module.Smodel import NoiseAda
from predictor.module.GNNs import GCN
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.tools import heatmap
import nni


class smodel_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'], norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'], output_layer=conf.model['output_layer']).to(self.device)
        self.noiseAda = NoiseAda(n_class=conf.model['n_classes'], noise_rate_init=conf.model['noise_rate_init']).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.noiseAda.parameters()),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, adj)
        pred = F.softmax(output, dim=1)
        eps = 1e-8
        score = self.noiseAda(pred).clamp(eps, 1 - eps)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = self.loss_fn(torch.log(score[mask]), label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc
