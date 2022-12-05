import torch
from torch.nn import Sigmoid


def get_state(output_batch, target_batch, thresh=0.5):
    output_batch = Sigmoid()(output_batch) >= thresh
    target_batch = target_batch.int().bool()
    tp = torch.sum(torch.logical_and(output_batch == True, target_batch == True))
    tn = torch.sum(torch.logical_and(output_batch == False, target_batch == False))
    fp = torch.sum(torch.logical_and(output_batch == True, target_batch == False))
    fn = torch.sum(torch.logical_and(output_batch == False, target_batch == True))

    return tp, tn, fp, fn


def precision(state_tuple):
    tp, _, fp, _ = state_tuple

    return div(tp, (tp + fp))


def recall(state_tuple):
    tp, _, _, fn = state_tuple

    return div(tp, (tp + fn))


def iou_score(state_tuple):
    tp, _, _, fn = state_tuple

    return div(tp, (tp + fn))


def accuracy(state_tuple):
    tp, tn, fp, fn = state_tuple

    return div((tp + tn), (tp + tn + fp + fn))


def f1_score(state_tuple):
    pr = precision(state_tuple)
    rc = recall(state_tuple)

    return div((2 * pr * rc), (pr + rc))


def div(x, y):
    if y == 0:
        return torch.tensor(0)
    else:
        return x / y
