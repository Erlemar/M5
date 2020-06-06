import torch
from torch.nn.modules.loss import _Loss


class CustomLossFunctions:
    """
    Custom loss functions for kpforecast.ml models.
    """

    @staticmethod
    def sMAPE(forecasted_out, target):
        """ sMAPE loss function

        Args:
            forecasted_out(torch.tensor): 2 dim tensor (2D array where the
                first dimension corresponds to the number of forecasts, and
                the second dimension corresponds to length of the forecast)
                containing a BATCH of forecasts.
            target(torch.tensor): 2 dim tensor with corresponing actuals/targets
                to the afformentioned forecasts

        Raises assertion error if forecasted_out and target dimensions/ranks/shapes aren't = 2
        Raises assertion error if the forecasted_out and target's shape are not the same
        """
        forecasted_dim = forecasted_out.shape
        target_shape = target.shape
        assert len(forecasted_dim) == len(target_shape) == 2
        assert forecasted_dim == target_shape
        smape = 0
        num_eval_samples = 0
        for idx, (targ, forc) in enumerate(zip(target, forecasted_out)):
            num_eval_samples += 1
            smape += torch.FloatTensor.abs(forc - targ) / (
                (torch.FloatTensor.abs(targ) + torch.FloatTensor.abs(forc)) / 2
            )
        smape = smape / num_eval_samples
        ret = torch.mean(smape) * 100

        return ret


def wrmsse(logits, labels, scales, weights):
    m = torch.mean(torch.pow((logits - labels), 2), axis=1)
    r = torch.sqrt(m / scales)
    return torch.sum(r * weights)


class WRMSSE(_Loss):
    def __init__(self, use_scale: bool = True, use_weights: bool = True):
        """
        Calculate wrmsse loss

        Args:
            use_scale: use scaling
            use_weights: use weights
        """
        super(WRMSSE, self).__init__()
        self.use_scale = use_scale
        self.use_weights = use_weights

    def forward(self, logits, labels, scales, weights):
        loss_value = torch.mean(torch.pow((logits - labels), 2), axis=1)
        if self.use_scale:
            loss_value = torch.sqrt(loss_value / scales)
        else:
            loss_value = torch.sqrt(loss_value)
        if self.use_weights:
            loss_value = torch.sum(loss_value * weights)
        else:
            loss_value = torch.sum(loss_value)
        return loss_value


class WMASSE(_Loss):
    def __init__(self, use_scale: bool = True, use_weights: bool = True):
        """
        Calculate wrmsse loss

        Args:
            use_scale: use scaling
            use_weights: use weights
        """
        super(WMASSE, self).__init__()
        self.use_scale = use_scale
        self.use_weights = use_weights

    def forward(self, logits, labels, scales, weights):
        loss_value = torch.mean(torch.abs(logits - labels), axis=1)
        if self.use_scale:
            loss_value = torch.sqrt(loss_value / scales)
        else:
            loss_value = torch.sqrt(loss_value)
        if self.use_weights:
            loss_value = torch.sum(loss_value * weights)
        else:
            loss_value = torch.sum(loss_value)
        return loss_value
