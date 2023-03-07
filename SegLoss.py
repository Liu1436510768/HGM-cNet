# 损失函数

import torch.nn as nn
import torch
import math
import torch.nn.functional as F


# log_loss
def log_loss(y, p):
    """
    --- Log_loss computing function
    A function to compute the log loss of a predicted probability p given
    a true target y.

    :param y: True target value
    :param p: Predicted probability
    :return: Log loss.
    """
    p = min(max(p, 10e-15), 1. - 10e-15)
    return -math.log(p) if y == 1 else -math.log(1. - p)


# Cross-Entropy Loss（多分类）用这个 loss 前面不需要加 Softmax 层
def CrossEntropyLoss(predict, target):
    loss = torch.nn.CrossEntropyLoss()(predict, target)
    return loss


# BCELoss(二分类)
def BCELoss(predict, target):
    loss = torch.nn.BCELoss()(predict, target)
    return loss


# MSELoss Mean Square Error
def MSELoss(predict, target):
    loss = torch.nn.MSELoss()(predict, target)
    return loss


# L1Loss Mean Absolute Error
def L1Loss(predict, target):
    loss = torch.nn.L1Loss()(predict, target)
    return loss


# BCEWithLogitsLoss
def BCEWithLogitsLoss(predict, target):
    loss = torch.nn.BCEWithLogitsLoss()(predict, target)
    return loss


# NLLLoss(多分类)
def NLLLoss(predict, target):
    loss = torch.nn.NLLLoss()(predict, target)
    return loss


# SmoothL1Loss(多分类)
def SmoothL1Loss(predict, target):
    loss = torch.nn.SmoothL1Loss()(predict, target)
    return loss


# DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# DiceMeanLoss
class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size()[1]

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num


# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
    ALPHA = 0.8
    GAMMA = 2

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


# --------------------------- BINARY LOSSES ---------------------------
class BFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, weight=None, ignore_index=255):
        super(BFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
# --------------------------- MULTICLASS LOSSES ---------------------------
class MFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, weight=None, ignore_index=255):
        super(MFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


# IoULoss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


def main():
    import numpy

    print(log_loss(0, 0))
    print(log_loss(0, 1))

    predict = torch.rand(2, 2, 24, 24, 48)  # predict是预测结果
    target = torch.empty(2, 2, 24, 24, 48).random_(0, 2)  # target是ground true

    print(BCEWithLogitsLoss(predict, target))

    predict = torch.sigmoid(predict)

    print(DiceLoss()(predict, target))
    print(DiceMeanLoss()(predict, target))
    print(FocalLoss()(predict, target))
    print(BFocalLoss()(predict, target))
    print(IoULoss()(predict, target))

    print(BCELoss(predict, target))
    print(MSELoss(predict, target))

    target = torch.empty(2, 24, 24, 48).random_(0, 2).long()  # target是ground true
    print(NLLLoss(predict, target))
    print(CrossEntropyLoss(predict, target))
    print(MFocalLoss()(predict, target))


if __name__ == '__main__':
    main()
