import torch

from src.models.n_beats.blocks.block import NBeatsBlock


class TrendBlock(NBeatsBlock):
    def __init__(self, f_b_dim, thetas_dim=(2, 2), num_hidden_layers=3, hidden_layer_dim=8, **kwargs):
        super(TrendBlock, self).__init__(f_b_dim, thetas_dim, num_hidden_layers, hidden_layer_dim)
        self.backcast_linspace, self.forecast_linspace = self.linspace()

    def forward(self, x):
        def trend_model(thetas, t, x):
            p = thetas.size()[-1]
            assert p <= 4, 'thetas_dim is too big.'
            T = torch.stack([t ** i for i in range(p)]).type_as(x)
            ret = thetas.mm(T)
            return ret

        thetas = super(TrendBlock, self).forward(x)
        backcast = trend_model(thetas[1], self.backcast_linspace, x)
        forecast = trend_model(thetas[0], self.forecast_linspace, x)
        return forecast, backcast

    def linspace(self):
        backcast_length = self._f_b_dim[1]
        forecast_length = self._f_b_dim[0]
        lin_space = torch.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
        b_ls = lin_space[:backcast_length]
        f_ls = lin_space[backcast_length:]
        return b_ls, f_ls
