import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import csv
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import time
from typing import Union


class MDatasetOld(Dataset):

    def __init__(self,
                 df: pd.DataFrame = None,
                 mode: str = 'train',
                 backcast_length: int = 7,
                 forecast_length: int = 7,
                 val_rate: float = 0.8):
        """

        Args:
            path: path to data

        """
        self.df = df.values
        self.mode = mode
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.val_rate = val_rate

    def __getitem__(self, idx):
        item_data = self.df[idx, 6:].reshape(-1, )
        if self.mode == 'train':
            min_ind = self.backcast_length
            max_ind = (len(item_data) - self.forecast_length - self.backcast_length + 1) * self.val_rate
        elif self.mode == 'valid':
            min_ind = (len(item_data) - self.forecast_length - self.backcast_length + 1) * self.val_rate
            max_ind = len(item_data) + 1 - self.forecast_length

        # rand_ind = np.random.randint(self.backcast_length, len(item_data) + 1 - self.forecast_length)
        rand_ind = np.random.randint(min_ind, max_ind)
        # print(rand_ind, idx)
        x = item_data[rand_ind - self.backcast_length: rand_ind].astype(float)
        # print('x', x.shape)
        y = item_data[rand_ind:rand_ind + self.forecast_length].astype(float)
        # print('y', y.shape)

        return x, y

    def __len__(self):
        return len(self.df)


# class MDataset(Dataset):
#
#     def __init__(self,
#                  df: pd.DataFrame = None,
#                  mode: str = 'train',
#                  backcast_length: int = 7,
#                  forecast_length: int = 7,
#                  train_history_modifier: float = 1.5):
#         """
#
#         Args:
#             path: path to data
#
#         """
#         self.df = df.values
#         self.mode = mode
#         self.backcast_length = backcast_length
#         self.forecast_length = forecast_length
#         self.train_history_modifier = train_history_modifier
#
#     def __getitem__(self, idx):
#         item_name = self.df[idx, 0]
#         item_data = self.df[idx, 1:].reshape(-1, )
#         if self.mode == 'train':
#             min_ind = len(item_data) - self.forecast_length * (1 + self.train_history_modifier) + 1
#             max_ind = len(item_data) - self.forecast_length + 1
#             rand_ind = np.random.randint(min_ind, max_ind)
#         elif self.mode == 'valid':
#             # min_ind = len(item_data) - self.forecast_length
#             # max_ind = len(item_data) + 1 - self.forecast_length
#             rand_ind = self.backcast_length
#
#         # rand_ind = np.random.randint(self.backcast_length, len(item_data) + 1 - self.forecast_length)
#
#         # print(rand_ind, idx)
#         x = item_data[rand_ind - self.backcast_length: rand_ind].astype(float)
#         # print('x', x.shape)
#         y = item_data[rand_ind:rand_ind + self.forecast_length].astype(float)
#         # print('y', y.shape)
#
#         return x, y, item_name
#
#     def __len__(self):
#         return len(self.df)

#
# class RMSELoss(torch.nn.Module):
#     def __init__(self):
#         super(RMSELoss,self).__init__()
#
#     def forward(self,x,y):
#         criterion = nn.MSELoss()
#         loss = torch.sqrt(criterion(x, y))
#         return loss
#
#
# # plot utils.
# def plot_scatter(*args, **kwargs):
#     plt.plot(*args, **kwargs)
#     plt.scatter(*args, **kwargs)
#
#
# # simple batcher.
# def data_generator(x_full, y_full, bs):
#     def split(arr, size):
#         arrays = []
#         while len(arr) > size:
#             slice_ = arr[:size]
#             arrays.append(slice_)
#             arr = arr[size:]
#         arrays.append(arr)
#         return arrays
#
#     while True:
#         for rr in split((x_full, y_full), bs):
#             yield rr
#
#
# def train_100_grad_steps_old(data, device, net, optimiser, test_losses, checkpoint_name='nbeats-training-checkpoint.th'):
#     global_step = load(net, optimiser, checkpoint_name)
#     criterion = RMSELoss()
#     for x_train_batch, y_train_batch in data:
#         global_step += 1
#         optimiser.zero_grad()
#         net.train()
#         _, forecast = net(torch.as_tensor(x_train_batch, dtype=torch.float).to(device))
#         # loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device))
#         loss = criterion(forecast, torch.as_tensor(y_train_batch, dtype=torch.float).to(device))
#         loss.backward()
#         optimiser.step()
#         if global_step % 100 == 0:
#             print(f'grad_step = {str(global_step).zfill(6)}, tr_loss = {loss.item():.6f}, te_loss = {test_losses[-1]:.6f}')
#         if global_step > 0 and global_step % 100 == 0:
#             with torch.no_grad():
#                 save(net, optimiser, global_step, checkpoint_name)
#             break
#
#     return net


def wrmsse(logits, labels, ws_dict, names, device):
    names = ['_'.join(j.split('_')[:-1])[:-5] + '--' + '_'.join(j.split('_')[:-1])[-4:] for j in names]
    m = torch.mean(torch.pow((logits.to(device) - labels), 2), axis=1)
    scales = torch.tensor([ws_dict['scale'][k] for k in names]).to(device)
    r = torch.sqrt(m / scales)
    weights = torch.tensor([ws_dict['weights'][k] for k in names]).to(device)
    return torch.sum(r * weights)


def train_100_grad_steps(train_dl, device, net, optimiser, test_losses, checkpoint_name='nbeats-training-checkpoint.th',
                         ws_dict=None):
    # criterion = RMSELoss()
    criterion = wrmsse
    total_loss = 0
    for step, (x_train_batch, y_train_batch, item_names) in enumerate(train_dl):
        optimiser.zero_grad()
        net.train()
        # _, forecast = net(torch.as_tensor(x_train_batch, dtype=torch.float).to(device))
        _, forecast = net(x_train_batch.float().to(device))
        # loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device))
        # loss = criterion(forecast, torch.as_tensor(y_train_batch, dtype=torch.float).to(device))
        # loss = criterion(forecast.float(), y_train_batch.float().to(device))
        loss = criterion(forecast.float(), y_train_batch.float().to(device), ws_dict, item_names, device)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    # print(f'Train loss: {total_loss}')
    # print(f'tr_loss = {loss.item():.6f}, te_loss = {test_losses[-1]:.6f}')
    with torch.no_grad():
        save(net, optimiser, 0, checkpoint_name)

    return net, total_loss


def load(model, optimiser, checkpoint_name):
    if os.path.exists(checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {checkpoint_name}.')
        return grad_step
    return 0


def save(model, optimiser, grad_step, checkpoint_name):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, checkpoint_name)


# evaluate model on test data and produce some plots.
def eval_test_old(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test):
    net.eval()
    criterion = RMSELoss()
    _, forecast = net(torch.as_tensor(x_test, dtype=torch.float))
    # test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
    test_losses.append(criterion(torch.as_tensor(forecast, dtype=torch.float), torch.as_tensor(y_test, dtype=torch.float)).item())
    p = forecast.cpu().detach().numpy()
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    plt.show()

    return test_losses


def eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, valid_loader, ws_dict=None):
    net.eval()
    criterion = RMSELoss()
    criterion = wrmsse
    y_true = []
    y_pred = []
    names = []
    for step, (x_valid_batch, y_valid_batch, item_names) in enumerate(valid_loader):
        # _, forecast = net(torch.as_tensor(x_valid_batch, dtype=torch.float))
        _, forecast = net(x_valid_batch.float())
        y_true.extend(y_valid_batch.cpu().detach().numpy())
        y_pred.extend(forecast.cpu().detach().numpy())
        names.extend(item_names)
    # test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
    # test_losses.append(criterion(torch.as_tensor(y_pred, dtype=torch.float), torch.as_tensor(y_true, dtype=torch.float)).item())
    # print(len(y_pred), len(y_true), len(names))
    loss = criterion(torch.as_tensor(y_pred, dtype=torch.float), torch.as_tensor(y_true, dtype=torch.float), ws_dict, names, 'cpu')
    test_losses.append(loss)
    # test_losses.append(criterion(torch.as_tensor(y_pred, dtype=torch.float), torch.as_tensor(y_true, dtype=torch.float)).item())
    p = forecast.cpu().detach().numpy()
    # subplots = [221, 222, 223, 224]
    # plt.figure(1)
    # for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
    #     ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
    #     plt.subplot(subplots[plot_id])
    #     plt.grid()
    #     plot_scatter(range(0, backcast_length), xx, color='b')
    #     plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
    #     plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    # plt.show()
    return test_losses


def get_m4_data(backcast_length, forecast_length, is_training=True):
    # https://www.mcompetitions.unic.ac.cy/the-dataset/

    if is_training:
        filename = 'data/Daily-train.csv'
    else:
        filename = 'data/m4/val/Daily-test.csv'

    x = np.array([]).reshape(0, backcast_length)
    y = np.array([]).reshape(0, forecast_length)
    x_tl = []
    headers = True
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
    x_tl_tl = np.array(x_tl)
    for i in range(x_tl_tl.shape[0]):
        if len(x_tl_tl[i]) < backcast_length + forecast_length:
            continue
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        if is_training:
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            j = np.random.randint(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length)
            time_series_cleaned_forlearning_x[0, :] = time_series_cleaned[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0, :] = time_series_cleaned[j:j + forecast_length]
        else:
            time_series_cleaned_forlearning_x = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
            time_series_cleaned_forlearning_y = np.zeros(
                (time_series_cleaned.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
            for j in range(backcast_length, time_series_cleaned.shape[0] + 1 - forecast_length):
                time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series_cleaned[j - backcast_length:j]
                time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series_cleaned[j: j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

    return x, y


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break

def train_model(model, train_dl, optimizer, loss_fn, epochs=1, val_dl = None, verbose=False, scheduler = None, metric_fns = {}, gradient_accumulation_steps=1, fp16=False, callbacks = None, hist = None, batch_unravel_fn = None):


    if hist is None:
        hist = {'metrics':{m:[] for m in metric_fns},'tr_loss':[],'val_loss':[],'tr_time':[],'val_time':[],'lr':[]}


    for epoch in range(epochs):

        start_time = time.time()

        tr_loss = 0
        model.train()
        for step, batch in tqdm(enumerate(train_dl),total=len(train_dl), disable=1-verbose):

            if batch_unravel_fn is None:
                batch = tuple(t.cuda() for t in batch)
                x, y, idx = batch
            else:
                x, y, idx = batch_unravel_fn(batch)

            logits = model(x)
            loss = loss_fn(logits, y)

            # if args['gradient_accumulation_steps'] > 1:
            #     loss = loss / args['gradient_accumulation_steps']

            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            tr_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
            print(scheduler.get_lr())
            hist['lr'] = scheduler.get_lr()
        tr_loss = tr_loss / len(train_dl)
        tr_time = time.time() - start_time

        hist['tr_loss'] += [tr_loss]
        hist['tr_time'] += [tr_time]

        # EVAL
        if val_dl is not None:
            val_start_time = time.time()
            val_hist = eval(val_dl, model, loss_fn, metric_fns=metric_fns,batch_unravel_fn = batch_unravel_fn)

            val_time = time.time()-val_start_time

            hist['val_loss'] += [val_hist['val_loss']]
            hist['val_time'] += [val_time]
            for m in val_hist['metrics']:
                hist['metrics'][m] += [val_hist['metrics'][m]]

        #apply callbacks
        for callback in callbacks:
            callback(epoch, hist, model)


class WRMSSEEvaluator(object):
    group_ids = ('all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
                 ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'],
                 ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 calendar: pd.DataFrame,
                 prices: pd.DataFrame):
        """
        intialize and calculate weights
        """
        self.calendar = calendar
        self.prices = prices
        self.train_df = train_df
        self.valid_df = valid_df
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()

        self.train_df['all_id'] = "all"

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],
                                      axis=1,
                                      sort=False)
        self.train_series = self.trans_30490_to_42840(self.train_df,
                                                      self.train_target_columns,
                                                      self.group_ids)
        self.valid_series = self.trans_30490_to_42840(self.valid_df,
                                                      self.valid_target_columns,
                                                      self.group_ids)
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        # self.train_series = None
        self.train_df = None
        self.prices = None
        self.calendar = None

    def get_scale(self):
        """
        scaling factor for each series ignoring starting zeros
        """
        scales = []
        for i in tqdm(range(len(self.train_series))):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series != 0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)

    @staticmethod
    def get_name(i):
        """
        convert a str or list of strings to unique string
        used for naming each of 42840 series
        """
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)

    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(
            ["item_id", "store_id"]
        )
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            "value"
        ]
        weight_df = weight_df.loc[
                    zip(self.train_df.item_id, self.train_df.store_id), :
                    ].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False
        )
        weights_map = {}
        for j, group_id in enumerate(tqdm(self.group_ids, leave=False)):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array(
                    [lv_weight.iloc[i]]
                )
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        """
        transform 30490 sries to all 42840 series
        """
        series_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T

    def get_rmsse(self, valid_preds) -> pd.Series:
        """
        returns rmsse scores for all 42840 series
        """
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],
                                axis=1,
                                sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds,
                                                self.valid_target_columns,
                                                self.group_ids,
                                                True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse],
                                      axis=1,
                                      sort=False).prod(axis=1)
        return np.sum(self.contributors)