import json
import pickle
from functools import lru_cache

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.get_dataset import get_datasets
from src.utils.get_model import get_m5model
from src.utils.utils import load_obj, flatten_omegaconf


class LitM5NBeats(pl.LightningModule):

    def __init__(self, hparams: DictConfig = None):
        super(LitM5NBeats, self).__init__()
        self.cfg = hparams
        self.hparams = flatten_omegaconf(hparams)
        self.net = get_m5model(self.cfg)
        self.hparams['n_params'] = sum(p.numel() for p in self.net.parameters())
        self.criterion = load_obj(self.cfg.loss.class_name)(**self.cfg.loss.params)
        self._prepare_evaluator()

    def _prepare_evaluator(self):
        with open(f'{self.cfg.data.folder_path}/saved_objects/evaluator.pickle', 'rb') as f:
            self.evaluator = pickle.load(f)

        ws = self.evaluator.weights.copy()
        ws.columns = ['weights']
        ws['scale'] = self.evaluator.scale
        with open(f'{self.cfg.data.folder_path}/saved_objects/name_mapping.json', 'r') as f:
            name_mapping = json.load(f)

        ws.index = [name_mapping[k] for k in ws.index]
        self.ws_dict = ws.to_dict()

    def forward(self, x, *args, **kwargs):
        return self.net(x)

    def prepare_data(self):
        datasets = get_datasets(self.cfg)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.cfg.data.batch_size,
                                                   num_workers=self.cfg.data.num_workers,
                                                   shuffle=True)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.cfg.data.batch_size,
                                                   num_workers=0)
        return valid_loader

    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    # def configure_optimizers(self):
    #     optimizer = load_obj(self.cfg.optimizer.class_name)(self.net.parameters(), **self.cfg.optimizer.params)
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #                 optimizer,
    #                 num_warmup_steps=self.hparams.warmup_steps,
    #                 num_training_steps=self.total_steps(),
    #     )
    #     return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(self.net.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return [optimizer], [{"scheduler": scheduler, "interval": self.cfg.scheduler.step}]

    def training_step(self, batch, batch_idx):
        x_train_batch, y_train_batch, scales, weights = batch

        forecast, loss = self.net(x_train_batch.float(),
                                  y_train_batch.float(),
                                  scales,
                                  weights)

        logs = {'loss': loss}
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x_valid_batch, y_valid_batch, scales, weights = batch
        forecast, loss = self.net(x_valid_batch.float(),
                                  y_valid_batch.float(),
                                  scales,
                                  weights)

        return {'val_loss': loss, 'y_pred': forecast}

    def validation_epoch_end(self, outputs):
        y_pred = []
        losses = []
        for step, x in enumerate(outputs):
            y_pred.extend(x['y_pred'].cpu().detach().numpy())
            losses.extend(x['val_loss'].cpu().detach().numpy())
        main_score = self.evaluator.score(np.array(y_pred))
        tensorboard_logs = {'main_score': main_score}
        return {'val_loss': np.sum(losses), 'main_score': main_score,
                'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
