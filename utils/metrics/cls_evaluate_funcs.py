#!/usr/bin/env python3 
# -*- coding: utf-8 -*-


import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, recall_score, precision_score


def cal_f1_score(pcs, rec):
    tmp = 2 * pcs * rec / (pcs + rec)
    return round(tmp, 4)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }



def f1_measures(preds, labels):
    f1 = f1_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    return {
        "f1": f1,
        "recall": recall,
        "precision": precision
    }


def acc(preds, labels):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
    }
