import pandas as pd
from omegaconf import DictConfig

from src.utils.utils import load_obj


def get_datasets(cfg: DictConfig) -> dict:
    """
    Get datases for modelling

    Args:
        cfg: config

    Returns:
        dict with datasets
    """
    train_df = pd.read_csv(f'{cfg.data.folder_path}/data/sales_train_evaluation.csv')

    train_fold_df = train_df.iloc[:, :-28]
    valid_fold_df = train_df.iloc[:, -28:].copy()

    backcast_length = cfg.dataset.backcast_length
    # forecast_length = cfg.dataset.forecast_length

    # train dataset
    dataset_class = load_obj(cfg.dataset.class_name)
    # get only useful columns
    train_data = pd.concat([train_fold_df.iloc[:, 0], train_fold_df.iloc[:, 6:-1]], axis=1)
    train_dataset = dataset_class(df=train_data, mode='train', cfg=cfg)

    valid_data = pd.concat(
        [train_fold_df.iloc[:, 0], train_fold_df.iloc[:, -backcast_length - 1 : -1], valid_fold_df], axis=1
    )
    valid_dataset = dataset_class(df=valid_data, mode='valid', cfg=cfg)

    return {'train': train_dataset, 'valid': valid_dataset}
