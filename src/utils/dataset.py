import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json
from omegaconf import DictConfig
from src.utils.metrics import WRMSSEEvaluator
import pickle


class DatasetTS(Dataset):
    """ Data Set Utility for Time Series.
    
        Args:
            - time_series(numpy 1d array) - array with univariate time series
            - forecast_length(int) - length of forecast window
            - backcast_length(int) - length of backcast window
            - sliding_window_coef(int) - determines how much to adjust sliding window
                by when determining forecast backcast pairs:
                    if sliding_window_coef = 1, this will make it so that backcast 
                    windows will be sampled that don't overlap. 
                    If sliding_window_coef=2, then backcast windows will overlap 
                    by 1/2 of their data. This creates a dataset with more training
                    samples, but can potentially lead to overfitting.
    """
    def __init__(self, time_series, forecast_length, backcast_length, sliding_window_coef=1):
        self.data = time_series
        self.forecast_length, self.backcast_length = forecast_length, backcast_length
        self.sliding_window_coef = sliding_window_coef
        self.sliding_window = int(np.ceil(self.backcast_length / sliding_window_coef))
    
    def __len__(self):
        """ Return the number of backcast/forecast pairs in the dataset.
        """
        length = int(np.floor((len(self.data)-(self.forecast_length+self.backcast_length)) / self.sliding_window))
        return length

    def __getitem__(self, index):
        """Get a single forecast/backcast pair by index.
            
            Args:
                index(int) - index of forecast/backcast pair
            raise exception if the index is greater than DatasetTS.__len__()
        """
        if(index > self.__len__()):
            raise IndexError("Index out of Bounds")
        # index = index * self.backcast_length
        index = index * self.sliding_window
        if index+self.backcast_length:
            backcast_model_input = self.data[index:index+self.backcast_length]
        else: 
            backcast_model_input = self.data[index:]
        forecast_actuals_idx = index+self.backcast_length
        forecast_actuals_output = self.data[forecast_actuals_idx:
                                            forecast_actuals_idx+self.forecast_length]
        forecast_actuals_output = np.array(forecast_actuals_output, dtype=np.float32)
        backcast_model_input = np.array(backcast_model_input, dtype=np.float32)
        return backcast_model_input, forecast_actuals_output


class M5NBeatsDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame = None,
                 mode: str = 'train',
                 cfg: DictConfig = None):
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
        item_data = self.df[idx, 1:].reshape(-1, )

        if self.mode == 'train':
            min_ind = len(item_data) - self.forecast_length * (1 + self.train_history_modifier) + 1
            max_ind = len(item_data) - self.forecast_length + 1
            rand_ind = np.random.randint(min_ind, max_ind)
        elif self.mode == 'valid':
            rand_ind = self.backcast_length

        x = item_data[rand_ind - self.backcast_length: rand_ind].astype(float)
        y = item_data[rand_ind:rand_ind + self.forecast_length].astype(float)

        scale = self.ws_dict['scale'][item_name]
        weight = self.ws_dict['weights'][item_name]

        return x, y, np.array(scale), np.array(weight)

    def __len__(self):
        return len(self.df)
