import copy
import datetime
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.n_beats.blocks.generic_block import GenericNBeatsBlock
from src.models.n_beats.blocks.seasonality_block import SeasonalityBlock
from src.models.n_beats.blocks.trend_block import TrendBlock
from src.models.n_beats.n_beats import NBeats
from src.models.n_beats_model import NBeatsNet
from src.utils.dataset import M5NBeatsDataset
from src.utils.metrics import WRMSSEEvaluator
from src.utils.utils import load_model, save_model, train_epoch, eval_test
from src.utils.custom_loss_functions import wrmsse
import pickle
train_df = pd.read_csv('data/sales_train_validation.csv')
calendar = pd.read_csv('data/calendar.csv')
prices = pd.read_csv('data/sell_prices.csv')

train_fold_df = train_df.iloc[:, :-28]
valid_fold_df = train_df.iloc[:, -28:].copy()

# e = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
with open('saved_objects/evaluator.pickle', 'rb') as f:
    e = pickle.load(f)

ws = e.weights.copy()
ws.columns = ['weights']
ws['scale'] = e.scale
ws_dict = ws.to_dict()

device = torch.device('cuda:0')  # use the trainer.py to run on GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forecast_length = 28
backcast_length = 56
batch_size = 1024
lr = 0.001
train_history_modifier=3.5
epochs = 5

f_b_dim = (forecast_length, backcast_length)
# f_b_dim = (backcast_length, forecast_length)

train_data = pd.concat([train_fold_df.iloc[:, 0], train_fold_df.iloc[:, 6:-1]], axis=1)
dataset = M5NBeatsDataset(train_data, 'train', backcast_length, forecast_length, train_history_modifier)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

valid_data = pd.concat([train_fold_df.iloc[:, 0], train_fold_df.iloc[:, -backcast_length-1: -1], valid_fold_df], axis=1)
valid_dataset = M5NBeatsDataset(valid_data, 'valid', backcast_length, forecast_length)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)


net = NBeats(stacks=[TrendBlock, SeasonalityBlock, GenericNBeatsBlock],
             f_b_dim=f_b_dim,
             num_blocks_per_stack=[2, 4, 4],
             thetas_dims=[[2, 2], [8, 8], [2, 8]],
             hidden_layer_dim=64,
             device=device)
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)
print('--- Training ---')
# initial_epoch_step = load_model(net, optimizer)
RUN_NAME = str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
TENSORBOARD_ENABLE = True
criterion = wrmsse
writer = SummaryWriter("logs/" + RUN_NAME) if TENSORBOARD_ENABLE else None

train_losses = []
test_losses = []
scores = []
epochs = 5
patience = 40
p = 0
best_score = 100
best_model = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.8)

for epoch in range(epochs):
    print(f'Time: {time.ctime()}. Epoch: {epoch}.')
    net, train_loss = train_epoch(train_loader, device, net, optimizer, ws_dict=ws_dict, criterion=criterion)
    test_loss, kaggle_score = eval_test(valid_loader, device, net, ws_dict=ws_dict, criterion=criterion, evaluator=e)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step(test_loss)

    scores.append(kaggle_score)

    if writer:
        writer.add_scalar("loss/training_loss", train_loss, epoch)
        writer.add_scalar("loss/validation_loss", test_loss, epoch)
        writer.add_scalar("loss/metric", kaggle_score, epoch)

    if kaggle_score <= best_score:
        best_score = kaggle_score
        p = 0
        best_model = copy.deepcopy(net)
    else:
        p += 1
        if p > patience:
            print('Stopping')
            break

    with torch.no_grad():
        save_model(net, optimizer, epoch, run_name=RUN_NAME)
