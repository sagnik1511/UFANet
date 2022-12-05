import os
import time
import torch
import numpy as np
from termcolor import cprint
from ..utils.training import (load_model_on_device,
                                  model_checkpoint,
                                  prepare_metric_scores,
                                  save_metric_curves,
                                  save_metric_data)
from ..utils.metrics import get_state


def run_single_patch(model,
                     real_patch,
                     mask_patch,
                     loss_fn,
                     metrics,
                     device,
                     segm_threshold):
    if device != "cpu":
        real_patch = real_patch.cuda()
        mask_patch = mask_patch.cuda()
    output = model(real_patch)
    state_dict = get_state(output_batch=output,
                           target_batch=mask_patch,
                           thresh=segm_threshold)
    metric_dict = dict()
    for name, metric in metrics.items():
        metric_dict[name] = metric(state_dict).item()
    return loss_fn(output, mask_patch), metric_dict


def train_single_epoch(model,
                       train_loader,
                       loss_fn,
                       optim,
                       metrics,
                       device,
                       track_result_counter,
                       segm_threshold):
    loss_array = []
    additional_metric_dict = dict()
    for metric in metrics:
        additional_metric_dict[metric] = []
    model.train()
    for index, (real_patch, mask_patch) in enumerate(train_loader):
        loss, metric_results = run_single_patch(model=model,
                                                real_patch=real_patch,
                                                mask_patch=mask_patch,
                                                loss_fn=loss_fn,
                                                metrics=metrics,
                                                device=device,
                                                segm_threshold=segm_threshold)
        if index % track_result_counter == 0:
            print(f"Step {index + 1} : ")
            for metric_name, result in metric_results.items():
                print(f"{metric_name} : {'%.9f'%result}", end=" ")
            print("\n")
        loss_array.append(loss.item())
        for metric_name, result in metric_results.items():
            additional_metric_dict[metric_name].append(result)
        optim.zero_grad()
        loss.backward()
        optim.step()

    return (model, optim), loss_array, additional_metric_dict


def validate_single_epoch(model,
                          val_loader,
                          loss_fn,
                          metrics,
                          device,
                          segm_threshold):
    loss_array = []
    additional_metric_dict = dict()
    for metric in metrics:
        additional_metric_dict[metric] = []
    model.eval()
    with torch.no_grad():
        for index, (real_patch, mask_patch) in enumerate(val_loader):
            loss, metric_results = run_single_patch(model=model,
                                                    real_patch=real_patch,
                                                    mask_patch=mask_patch,
                                                    loss_fn=loss_fn,
                                                    metrics=metrics,
                                                    device=device,
                                                    segm_threshold=segm_threshold)
            loss_array.append(loss.item())
            for metric_name, result in metric_results.items():
                additional_metric_dict[metric_name].append(result)

    return loss_array, additional_metric_dict


def train_model(model,
                train_loader,
                val_loader,
                num_epochs,
                loss_fn,
                optim,
                metrics,
                device,
                track_result_counter,
                segm_threshold,
                store_result_dir="results",
                train_kill_thresh=5,
                figure_shape=(50, 15)):
    cprint("Training initiated...", "blue")
    init = time.time()
    model = load_model_on_device(model, device)
    train_op_dir = os.path.join(store_result_dir, "train")
    val_op_dir = os.path.join(store_result_dir, "val")
    best_loss = np.inf
    train_loss_array = []
    val_loss_array = []
    train_metric_result_dict = dict()
    val_metric_result_dict = dict()
    degradation_flag = 0
    for metric_name, _ in metrics.items():
        train_metric_result_dict[metric_name] = []
        val_metric_result_dict[metric_name] = []
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} :")
        epoch_init = time.time()
        (model, optim),\
        train_loss_array_single_epoch,\
        train_metric_result_dict_single_epoch = train_single_epoch(model=model,
                                                                   train_loader=train_loader,
                                                                   loss_fn=loss_fn,
                                                                   optim=optim,
                                                                   metrics=metrics,
                                                                   device=device,
                                                                   track_result_counter=track_result_counter,
                                                                   segm_threshold=segm_threshold)
        for metric_name, metric_arr in train_metric_result_dict_single_epoch.items():
            train_metric_result_dict[metric_name].append(np.mean(metric_arr))
        train_loss_array.append(np.mean(train_loss_array_single_epoch))
        train_epoch_loss = sum(train_loss_array_single_epoch)
        print(f"Training Loss : {'%.6f'%train_epoch_loss}")
        prepare_metric_scores(result_dict=train_metric_result_dict_single_epoch,
                              op_dir=train_op_dir,
                              figure_shape=figure_shape,
                              epoch=epoch)
        model_checkpoint(os.path.join(train_op_dir, str(epoch)), model, optim, epoch)
        val_loss_array_single_epoch, val_metric_result_dict_single_epoch = validate_single_epoch(model=model,
                                                                                                 val_loader=val_loader,
                                                                                                 loss_fn=loss_fn,
                                                                                                 metrics=metrics,
                                                                                                 device=device,
                                                                                                 segm_threshold=segm_threshold)
        for metric_name, metric_arr in val_metric_result_dict_single_epoch.items():
            val_metric_result_dict[metric_name].append(np.mean(metric_arr))
        val_loss_array.append(np.mean(val_loss_array_single_epoch))
        val_epoch_loss = sum(val_loss_array_single_epoch)
        print(f"Validation Loss : {'%.6f' % val_epoch_loss}")
        prepare_metric_scores(result_dict=val_metric_result_dict_single_epoch,
                              op_dir=val_op_dir,
                              figure_shape=figure_shape,
                              epoch=epoch)
        if best_loss > val_epoch_loss:
            cprint("Model Update : POSITIVE", "green")
            best_loss = val_epoch_loss
            degradation_flag = 0
            model_checkpoint(directory=store_result_dir,
                             model=model,
                             optim=optim)
        else:
            cprint("Model Update : NEGATIVE", "red")
            degradation_flag += 1
            if degradation_flag == train_kill_thresh:
                cprint("Training stopped due to continuous degraded learning...", "red")
                break
        cprint(f"Epoch execution Time : {'%.3f'%(time.time() - epoch_init)} seconds\n", "blue")
    for op_dir, res_dict in zip([train_op_dir, val_op_dir],
                                [train_metric_result_dict, val_metric_result_dict]):
        for metric_name, metric_arr in res_dict.items():
            save_metric_curves(value_arr=metric_arr,
                               directory=op_dir,
                               name=metric_name,
                               figure_shape=figure_shape,
                               epoch=None)
        save_metric_data(result_dict=res_dict, directory=op_dir)
    save_metric_curves(value_arr=train_loss_array,
                       directory=train_op_dir,
                       name="loss_fn",
                       figure_shape=figure_shape,
                       epoch=None)
    save_metric_curves(value_arr=val_loss_array,
                       directory=val_op_dir,
                       name="loss_fn",
                       figure_shape=figure_shape,
                       epoch=None)
    cprint("Training Finished...", "blue")
    cprint(f"Execution Time : {'%.3f'%(time.time() - init)} seconds", "blue")
