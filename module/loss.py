import torch.nn.functional as F
from metric import pearson
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output.squeeze(), target.squeeze())

def l2_loss(output, target):
    return F.mse_loss(output.squeeze(), target.squeeze())

def pearson_l2_loss(output, target):
    pearson_coef = pearson(output, target)
    mse = l2_loss(output, target)
    loss = (1 - pearson_coef) + mse
    return loss

def CE_l2_loss(output, target):
    bin_logits, logits = output
    bin_labels, labels = target['bin'], target['label']
    cls_loss = nn.CrossEntropyLoss()(bin_logits.squeeze(), bin_labels.squeeze())
    mse_loss = l2_loss(logits, labels)
    loss = cls_loss + mse_loss
    return loss