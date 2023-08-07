import torch.nn as nn
from torch_geometric.nn.models import GCN
import torch
import numpy as np
import random
import logging

torch.use_deterministic_algorithms(True)
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class LSTM_GCN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers_gcn):
        super(LSTM_GCN, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=False)
        self.gcn = GCN(in_channels=input_size, hidden_channels=input_size, num_layers=num_layers_gcn, normalize=False)

    @property
    def _device(self):
        assert next(self.lstm.parameters()).device == next(self.gcn.parameters()).device
        return next(self.lstm.parameters()).device
    
    def _h0(self, input_aux):
        return torch.zeros(1, input_aux[0].size()[0], input_aux[0].size()[1])

    def _c0(self, input_aux):
        return torch.zeros(1, input_aux[0].size()[0], input_aux[0].size()[1])
    
    def to_device(self, h0, c0, input_aux, edges):
        h0, c0, input_aux, edges = h0.to(self._device), c0.to(self._device), input_aux.to(self._device).detach(), edges.to(self._device).detach()
        return h0, c0, input_aux, edges

    def forward(self, input_aux, edges):
        h0, c0 = self._h0(input_aux), self._c0(input_aux)

        h0, c0, input_aux, edges = self.to_device(h0, c0, input_aux, edges)

        output, (hn, _) = self.lstm.forward(input_aux, (h0, c0))
        output = self.gcn.forward(torch.squeeze(hn), edges)
        return output

