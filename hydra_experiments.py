import hydra.experimental
from itertools import product
from hydra_run import run
from omegaconf import DictConfig
from typing import List, Dict
import argparse
import os
from src.utils.metrics import WRMSSEEvaluator
from src.utils.utils import product_dict, config_to_hydra_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hydra experiments for M5")
    parser.add_argument("--config_dir", help="main config dir", type=str, default="conf/")
    parser.add_argument("--experiment_config", help="experiment config", type=str, default="experiments.yaml")
    parser.add_argument("--main_config", help="main config", type=str, default="config.yaml")
    args = parser.parse_args()

    hydra.experimental.initialize(config_dir=args.config_dir, strict=True)

    experiment_cfg = hydra.experimental.compose(config_file="experiments.yaml")

    experiment_dict = config_to_hydra_dict(experiment_cfg)

    experiments = list(product_dict(**experiment_dict))

    for experiment in experiments:
        print(os.getcwd())
        cfg = hydra.experimental.compose(config_file="config.yaml", overrides=experiment)
        print(cfg.pretty())
        # run(cfg)
