from src.models.n_beats.blocks.block import NBeatsBlock

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class SeasonalityBlock(NBeatsBlock):
    def __init__(
        self,
        f_b_dim,
        thetas_dim=(8, 8),
        num_hidden_layers=2,
        hidden_layer_dim=1,
        layer_nonlinearity=nn.ReLU,
        layer_w_init=nn.init.xavier_uniform_,
        layer_b_init=nn.init.zeros_,
        shared_g_theta=None,
    ):

        super(SeasonalityBlock, self).__init__(f_b_dim, thetas_dim=thetas_dim, hidden_layer_dim=hidden_layer_dim)
        self.backcast_linspace, self.forecast_linspace, self.f_ls1 = self.linspace()

    def forward(self, x):
        def seasonality_model(self, thetas, t, x):
            p = thetas.size()[-1]
            assert p < 10, 'thetas_dim is too big.'
            p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
            s1 = torch.stack([np.cos(2 * np.pi * i * t) for i in range(p1)]).type_as(x)
            s2 = torch.stack([np.sin(2 * np.pi * i * t) for i in range(p2)]).type_as(x)
            S = torch.cat([s1, s2])
            return thetas.mm(S)

        thetas = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self, thetas[1], self.backcast_linspace, x)
        forecast = seasonality_model(self, thetas[0], self.forecast_linspace, x)
        forecast1 = seasonality_model(self, thetas[0], self.f_ls1, x)
        return forecast, backcast, forecast1

    def linspace(self):
        backcast_length = self._f_b_dim[1]
        forecast_length = self._f_b_dim[0]
        lin_space = torch.linspace(
            -backcast_length, forecast_length * 1.5, backcast_length + round(forecast_length * 1.5)
        )
        b_ls = lin_space[:backcast_length]
        f_ls = lin_space[backcast_length : backcast_length + forecast_length]
        f_ls1 = lin_space[backcast_length + forecast_length :]
        return b_ls, f_ls, f_ls1
