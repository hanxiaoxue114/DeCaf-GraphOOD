import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import time
import enlighten
import logging
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
# from utils import *
from typing import Callable, Iterator, Optional, Union
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

# cudaid = "cuda:0"
# device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
# def average_precision_score(y_true, y_pred, num_class):
#     ap = 0.
#     n = num_class
#     for c in range(num_class):
#         idx_pred_c = (y_pred == c).nonzero()
#         n_pred_c = idx_pred_c.size()[0]
#         n_true_c = (y_true[idx_pred_c] == c).sum()
#         if n_pred_c == 0:
#             precision = 0
#             if (y_true == c).sum()==0:
#                 n -= 1
#         else: precision = n_true_c/n_pred_c
#         ap += precision
#
#     ap = ap/n
#     return ap

def eval_acc(y_true, y_pred):
    acc_list = []
    if len(y_true.size()) == 1:
        y_true = y_true.unsqueeze(1)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        # is_labeled = y_true[:, i] == y_true[:, i]
        # correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        correct = y_true[:, i] == y_pred[:, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)*100

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    if len(y_true.size()) > 1 and y_true.size()[1] > 1:
        y_true = torch.argmax(y_true, 1)

    try:
        score = roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    except ValueError:
        score = 0
    return score*100

def accuracy(output, labels):
    num_class = labels.max() + 1
    if labels.max() > 1:
        preds = torch.argmax(output, dim=1).type_as(labels)
    else:
        preds = (output > 0.5).type_as(labels)
    # if len(labels.size()) > 1:
    #     labels = torch.argmax(labels, 1)
    if len(labels.size()) > 1 and labels.size()[1] == 1:
        labels = labels.squeeze(1)
    if len(labels.size()) > 1 and labels.size()[1] > 1:
        labels = torch.argmax(labels, 1)
    bacc = balanced_accuracy_score(labels.cpu(), preds.cpu())*100
    # precision = average_precision_score(labels.cpu(), preds.cpu(), num_class)*100
    f1 = f1_score(labels.cpu(), preds.cpu(),  labels=None, average='macro')*100
    if len(output.size()) <= 1 or output.size()[1]==1:
        acc_or_auc = eval_rocauc(labels, output)
    else:
        acc_or_auc = eval_acc(labels, output)
    try:
        ll = log_loss(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
    except ValueError:
        ll = 0
    return acc_or_auc, bacc, f1, ll

def get_activation(name: str, leaky_relu: Optional[float] = 0.5) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(leaky_relu)
    elif name == "rrelu":
        return nn.RReLU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "softmax":
        return nn.Softmax(dim=1)
    else:
        return None

class M_Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])

class G_Dataset(Dataset):
    def __init__(self, features, neighbors, labels):
        self.features = features
        self.neighbors = neighbors
        self.labels = labels

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, index):
        return (self.features[index], self.labels[index], self.neighbors[index])

def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)
