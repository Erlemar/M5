import pandas as pd
from omegaconf import DictConfig

from src.utils.utils import load_obj


def get_datasets(cfg: DictConfig):
    """
    Get datases for modelling

    Args:
        cfg: config

    Returns:

    """
    train_df = pd.read_csv(f'{cfg.data.folder_path}/data/sales_train_validation.csv')

    train_fold_df = train_df.iloc[:, :-28]
    valid_fold_df = train_df.iloc[:, -28:].copy()

    backcast_length = cfg.dataset.backcast_length
    forecast_length = cfg.dataset.forecast_length

    # train dataset
    dataset_class = load_obj(cfg.dataset.class_name)
    # get only useful columns
    train_data = pd.concat([train_fold_df.iloc[:, 0], train_fold_df.iloc[:, 6:-1]], axis=1)
    train_dataset = dataset_class(train_data,
                                  'train',
                                  cfg)

    valid_data = pd.concat([train_fold_df.iloc[:, 0],
                            train_fold_df.iloc[:, -backcast_length - 1: -1],
                            valid_fold_df],
                           axis=1)
    valid_dataset = dataset_class(valid_data,
                                  'valid',
                                  cfg)

    return {'train': train_dataset, 'valid': valid_dataset}
