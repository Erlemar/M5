import pandas as pd
from omegaconf import DictConfig
import torch
from src.utils.utils import load_obj


def get_m5model(cfg: DictConfig):
    """
    Get model

    Args:
        cfg: config

    Returns:

    """
    backcast_length = cfg.dataset.backcast_length
    forecast_length = cfg.dataset.forecast_length

    f_b_dim = (forecast_length, backcast_length)

    stacks = []
    num_blocks_per_stack = []
    thetas_dims = []
    for block in cfg.model.blocks:
        stacks.append(load_obj(block['stack']))
        num_blocks_per_stack.append(block['num_blocks_per_stack'])
        thetas_dims.append(block['thetas_dims'])

    net = load_obj(cfg.model.class_name)
    net = net(stacks=stacks,
              f_b_dim=f_b_dim,
              num_blocks_per_stack=num_blocks_per_stack,
              thetas_dims=thetas_dims,
              hidden_layer_dim=cfg.model.hidden_layer_dim)

    return net