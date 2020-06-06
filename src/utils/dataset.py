import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json
from omegaconf import DictConfig
from src.utils.metrics import WRMSSEEvaluator
import pickle


class M5NBeatsDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, mode: str = 'train', cfg: DictConfig = None):
        """
        Prepare data for nbeats model.

        Backcast - length of training sequence
        forecast - length of prediction
        train_history_modifier - determined the range to sample random index

        Args:
            df:
            mode:
            cfg
        """
        self.df = df.values
        self.mode = mode
        self.cfg = cfg
        self.backcast_length = cfg.dataset.backcast_length
        self.forecast_length = cfg.dataset.forecast_length
        self.train_history_modifier = cfg.dataset.train_history_modifier
        self._prepare_data()

    def _prepare_data(self):
        """
        Convert names to a usable format and get dict with scales and weights

        Returns:
            None
        """
        names = ['_'.join(j.split('_')[:-1])[:-5] + '--' + '_'.join(j.split('_')[:-1])[-4:] for j in self.df[:, 0]]
        self.df[:, 0] = names

        with open(f'{self.cfg.data.folder_path}/saved_objects/evaluator.pickle', 'rb') as f:
            evaluator = pickle.load(f)

        ws = evaluator.weights.copy()
        ws.columns = ['weights']
        ws['scale'] = evaluator.scale

        self.ws_dict = ws.to_dict()

    def __getitem__(self, idx):
        item_name = self.df[idx, 0]
        item_data = self.df[idx, 1:].reshape(-1)

        if self.mode == 'train':
            min_ind = len(item_data) - self.forecast_length * (1 + self.train_history_modifier) + 1
            max_ind = len(item_data) - self.forecast_length + 1
            rand_ind = np.random.randint(min_ind, max_ind)
        elif self.mode == 'valid':
            rand_ind = self.backcast_length

        x = item_data[rand_ind - self.backcast_length : rand_ind].astype(float)
        y = item_data[rand_ind : rand_ind + self.forecast_length].astype(float)

        scale = self.ws_dict['scale'][item_name]
        weight = self.ws_dict['weights'][item_name]

        return x, y, np.array(scale).reshape(1), np.array(weight).reshape(1)

    def __len__(self):
        return len(self.df)
