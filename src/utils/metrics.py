# Libraries
import math
from typing import Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as sk_mse
from tqdm import tqdm


class Metrics:
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        """Mean absolute percentage error.

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        result = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return result

    @staticmethod
    def root_mean_square_error(y_true: np.array, y_pred: np.array, n: int = 0) -> float:
        """Root mean squared error.

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        if n == 0:
            # This is directly from the sklearn metric
            mse = sk_mse(y_true, y_pred)
            result = math.sqrt(mse)
            return result
        else:
            # This is directly from the sklearn metric
            mse = sk_mse(y_true, y_pred, n)
            result = math.sqrt(mse)
            return result

    # TODO Understand why the need is there to overload this method?s
    @staticmethod
    def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
        """Root mean squared error (Overloaded from sklearn).

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        mse = sk_mse(y_true, y_pred)
        return mse

    @staticmethod
    def median_relative_absolute_error(y_true: np.array, y_pred: np.array, naive: np.array) -> float:
        """Median RAE (Overloaded from sklearn).

        Args:
            y_true -- truth value
            y_pred -- Predicted value
            n -- number of periods
        """
        result = []
        for _ in range(len(y_pred)):
            ii = np.abs(np.array(y_true) - np.array(y_pred)) / np.abs(np.array(naive) - np.array(y_true))
            if ii is None:
                result.append(1)
            else:
                result.append(ii)
        return np.ma.median(result)


class WRMSSEEvaluator(object):
    group_ids = (
        'all_id',
        'state_id',
        'store_id',
        'cat_id',
        'dept_id',
        'item_id',
        ['state_id', 'cat_id'],
        ['state_id', 'dept_id'],
        ['store_id', 'cat_id'],
        ['store_id', 'dept_id'],
        ['item_id', 'state_id'],
        ['item_id', 'store_id'],
    )

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        """
        Intialize and calculate weights
        """
        self.calendar = calendar
        self.prices = prices
        self.train_df = train_df
        self.valid_df = valid_df
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()

        self.train_df['all_id'] = 'all'

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        if not all(c in self.valid_df.columns for c in self.id_columns):
            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df], axis=1, sort=False)
        self.train_series = self.trans_30490_to_42840(self.train_df, self.train_target_columns, self.group_ids)
        self.valid_series = self.trans_30490_to_42840(self.valid_df, self.valid_target_columns, self.group_ids)
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        # self.train_series = None
        self.train_df = None
        self.prices = None
        self.calendar = None

    def get_scale(self) -> np.array:
        """
        Scaling factor for each series ignoring starting zeros
        """
        scales = []
        for i in tqdm(range(len(self.train_series))):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series != 0) :]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)

    @staticmethod
    def get_name(i: Any[str, int]) -> str:
        """
        Convert a str or list of strings to unique string
        used for naming each of 42840 series
        """
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return '--'.join(i)

    def get_weight_df(self) -> pd.DataFrame:
        """
        Return weights for each of 42840 series in a dataFrame
        """
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)
        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        weights_map = {}
        for _, group_id in enumerate(tqdm(self.group_ids, leave=False)):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array([lv_weight.iloc[i]])
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df: pd.DataFrame, cols: list, group_ids: Any, dis: bool = False) -> pd.DataFrame:
        """
        Transform 30490 sries to all 42840 series
        """
        series_map = {}
        for _, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T

    def get_rmsse(self, valid_preds: np.array) -> pd.Series:
        """
        Return rmsse scores for all 42840 series
        """
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds, self.valid_target_columns, self.group_ids, True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse], axis=1, sort=False).prod(axis=1)
        return np.sum(self.contributors)
