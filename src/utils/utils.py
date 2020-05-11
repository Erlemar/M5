import importlib
import os
import random
from itertools import product
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import shutil
import hydra
import collections


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def load_model(model, optimiser, checkpoint_name):
    if os.path.exists(checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Restored checkpoint from {checkpoint_name}.')
        return epoch
    return 0


def save_model(model, optimiser, epoch, run_name):
    path = "data/" + run_name + "/saved_models/"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, path + "model_epoch_" + str(epoch) + ".pt")


def train_epoch(train_dl, device, net, optimizer, ws_dict=None, criterion=None):
    total_loss = 0
    for step, (x_train_batch, y_train_batch, item_names) in enumerate(train_dl):
        optimizer.zero_grad()
        net.train()
        # _, forecast = net(x_train_batch.float().to(device))
        forecast, _ = net(x_train_batch.float().to(device))
        # print('forecast', forecast.shape)
        # print('y_train_batch', y_train_batch.shape)
        loss = criterion(forecast.float(), y_train_batch.float().to(device), ws_dict, item_names, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return net, total_loss


def eval_test(valid_loader, device, net, ws_dict=None, criterion=None, evaluator=None):
    net.eval()
    y_true = []
    y_pred = []
    names = []
    for step, (x_valid_batch, y_valid_batch, item_names) in enumerate(valid_loader):
        forecast, _ = net(x_valid_batch.float().to(device))
        y_true.extend(y_valid_batch.cpu().detach().numpy())
        y_pred.extend(forecast.cpu().detach().numpy())
        names.extend(item_names)
    # print('y_pred', np.array(y_pred).shape)
    main_score = evaluator.score(np.array(y_pred))

    loss = criterion(torch.as_tensor(y_pred, dtype=torch.float).to(device),
                     torch.as_tensor(y_true, dtype=torch.float).to(device), ws_dict, names, device)
    return loss, main_score


def set_seed(seed: int = 666, precision: int = 10):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)


def product_dict(**kwargs) -> List[List]:
    """
    Convert dict with lists in values into lists of all combinations

    This is necessary to convert config with experiment values
    into format usable by hydra
    Args:
        **kwargs:

    Returns:
        list of lists

    ---
    Example:
        >>> list_dict = {'a': [1, 2], 'b': [2, 3]}
        >>> list(product_dict(**list_dict))
        >>> [['a=1', 'b=2'], ['a=1', 'b=3'], ['a=2', 'b=2'], ['a=2', 'b=3']]

    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        zip_list = list(zip(keys, instance))
        yield [f'{i}={j}' for i, j in zip_list]


def config_to_hydra_dict(cfg: DictConfig) -> Dict:
    """
    Convert config into dict with lists of values, where key is full name of parameter

    This fuction is used to get key names which can be used in hydra.

    Args:
        cfg:

    Returns:

    """
    experiment_dict = {}
    for k, v in cfg.items():
        for k1, v1 in v.items():
            experiment_dict[f'{k}.{k1}'] = v1

    return experiment_dict


def save_useful_info():
    shutil.copytree(os.path.join(hydra.utils.get_original_cwd(), 'src'),
                    os.path.join(os.getcwd(), 'code/src'))
    shutil.copy2(os.path.join(hydra.utils.get_original_cwd(), 'hydra_run.py'), os.path.join(os.getcwd(), 'code'))


def flatten_omegaconf(d, sep="_"):

    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float]}
    # obj = {k: v for k, v in obj.items()}

    return obj
