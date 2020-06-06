import argparse
import glob
from typing import List

import hydra.experimental
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from src.lightning_classes.lightning_nbeats import LitM5NBeats
from src.utils.metrics import WRMSSEEvaluator
from src.utils.utils import set_seed
from src.utils.get_dataset import get_datasets


def make_prediction(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model inference

    Args:
        cfg: hydra config

    Returns:
        None
    """
    set_seed(cfg.training.seed)
    model_name = glob.glob(f'outputs/{cfg.inference.run_name}/saved_models/*')[0]

    lit_model = LitM5NBeats.load_from_checkpoint(checkpoint_path=model_name, cfg=cfg)

    net = lit_model.net

    datasets = get_datasets(cfg)

    loader = torch.utils.data.DataLoader(
        datasets[cfg.inference.mode], batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, shuffle=False
    )

    y_pred: List[np.array] = []
    device = cfg.data.device

    net.to(device)
    net.eval()

    for _, (x, y, scales, weights) in enumerate(loader):
        forecast, loss = net(x.float().to(device), y.float().to(device), scales.to(device), weights.to(device))
        y_pred.extend(forecast.cpu().detach().numpy())

    y_pred = np.array(y_pred)

    sub = pd.read_csv(f'{cfg.data.folder_path}/data/sample_submission.csv')

    sub.iloc[:30490, 1:] = y_pred
    sub.to_csv(f'subs/{cfg.inference.run_name}_{cfg.inference.mode}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference in M5 competition')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2020_05_16_15_43_39')
    parser.add_argument('--mode', help='valid or test', type=str, default='valid')
    args = parser.parse_args()

    hydra.experimental.initialize(config_dir='conf', strict=True)
    inference_cfg = hydra.experimental.compose(config_file='config.yaml')
    inference_cfg['inference']['run_name'] = args.run_name
    inference_cfg['inference']['mode'] = args.mode

    path = f'outputs/{inference_cfg.inference.run_name}/.hydra/config.yaml'

    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)

    cfg = OmegaConf.create(cfg_yaml)
    cfg['inference'] = inference_cfg.inference
    make_prediction(cfg)
