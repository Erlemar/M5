from typing import Dict, Any

from omegaconf import DictConfig

from src.utils.utils import load_obj


def get_m5model(cfg: DictConfig) -> Any:
    """
    Get model

    Args:
        cfg: config

    Returns:
        initialized model
    """
    backcast_length = cfg.dataset.backcast_length
    forecast_length = cfg.dataset.forecast_length

    f_b_dim = (forecast_length, backcast_length)

    # collect stack parameters.
    model_dict: Dict[str, list] = {k: [] for k in cfg.model.blocks[0].keys()}
    for block in cfg.model.blocks:
        for k, v in block.items():
            if type(v) == str:
                v = load_obj(v)
            model_dict[k].append(v)

    criterion = load_obj(cfg.loss.class_name)(**cfg.loss.params)
    net = load_obj(cfg.model.class_name)
    net = net(f_b_dim=f_b_dim, criterion=criterion, **model_dict)

    return net
