import os
import torch
import numpy as np
import pandas as pd
from termcolor import cprint
import matplotlib.pyplot as plt
from .metrics import *
from torch.utils.data import DataLoader
from ..data import UBIrisDataset, UBIrisDatasetTest


def prepare_metric_scores(result_dict, op_dir, figure_shape, epoch=None):
    for metric_name, metric_arr in result_dict.items():
        value = np.mean(metric_arr)
        print(f"{metric_name} : {'%.6f'%value}")
        save_metric_curves(value_arr=metric_arr,
                           directory=op_dir,
                           name=metric_name,
                           figure_shape=figure_shape,
                           epoch=epoch)
    print("\n")


def save_metric_curves(value_arr, directory, name, figure_shape, epoch):
    plt.figure(figsize=figure_shape)
    plt.plot(value_arr)
    title = f"Epoch_{epoch}_{name}" if epoch else name
    plt.title(title)
    dir_name = os.path.join(directory, str(epoch)) if epoch else directory
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    plt.savefig(os.path.join(dir_name, name))


def save_metric_data(result_dict, directory):
    dataframe = pd.DataFrame()
    filename = os.path.join(directory, "results.csv")
    for name, res_arr in result_dict.items():
        dataframe[name] = res_arr
    dataframe.to_csv(filename, index=False)
    print(f"Results have been stored in {filename} ...")


def load_model_on_device(model, device):
    dev = torch.device(device)
    model.to(dev)
    cprint("Model loaded on device...", "blue")
    return model


def model_checkpoint(directory, model, optim, epoch=None):
    filename = f"epoch_{epoch}.pt" if epoch else "best_model.pt"
    checkpoint = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
    }
    torch.save(checkpoint, os.path.join(directory, filename))


def fetch_metric_dict():
    metric_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou_score": iou_score,
    }
    return metric_dict


def create_data_loader(config, train=True, shuffle=True, drop_last=True, evaluate=False):
    if evaluate:
        ds = UBIrisDatasetTest(root_dir=config.DATA_DIRECTORY,
                               width=config.WIDTH)
    else:
        if train:
            metadata_base_path = os.path.join(config.DATA_DIRECTORY, config.TRAIN_METADATA_BASE_PATH)
        else:
            metadata_base_path = os.path.join(config.DATA_DIRECTORY, config.VAL_METADATA_BASE_PATH)
        ds = UBIrisDataset(meta_df_path=metadata_base_path,
                           real_width=config.REAL_WIDTH,
                           mask_width=config.MASK_WIDTH)
    return DataLoader(dataset=ds,
                      batch_size=config.BATCH_SIZE,
                      shuffle=config.SHUFFLE,
                      drop_last=config.DROP_LAST)
