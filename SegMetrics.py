# 分割计算指标

import torch
import numpy
from medpy import metric
from scipy.ndimage import morphology
import numpy as np


def dice1(logits, targets, class_index):
    smooth = 1e-5  # 防止0除
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + smooth) / (union + smooth)
    return dice


def diceMean(logits, targets):
    smooth = 1e-5  # 防止0除
    class_num = logits.size(1)

    dice_sum = 0
    for i in range(class_num):
        inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
        union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
        dice = (2. * inter + smooth) / (union + smooth)
        dice_sum += dice
    return dice_sum / class_num


def recall0(predict, target):  # Sensitivity, Recall, true positive rate都一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    fn = numpy.count_nonzero(~predict & target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


# 调包计算指标
def dice(pred, target):
    dice = metric.binary.dc(pred, target)
    return dice


def jaccard(pred, target):
    jaccard = metric.binary.jc(pred, target)
    return jaccard


def precision(pred, target):
    precision = metric.binary.precision(pred, target)
    return precision


def recall(pred, target):
    recall = metric.binary.recall(pred, target)
    return recall


def MeanDistance(pred, target):
    MD = metric.binary.asd(pred, target, voxelspacing=1, connectivity=18)
    # MD = numpy.mean(metric.binary.__surface_distances(pred, target, voxelspacing=1, connectivity=18))
    return MD


def Hausdorff(pred, target):
    HD = metric.binary.hd(pred, target)
    return HD


def Hausdorff95(pred, target):
    HD95 = metric.binary.hd95(pred, target)
    return HD95


def ASSD(pred, target):
    ASSD = metric.binary.assd(pred, target, voxelspacing=1, connectivity=18)
    return ASSD


def RMSD(pred, target):
    """
    计算 Root Mean Square symmetric Surface Distance
    """
    residual_mean_square_distance = metric.binary.__surface_distances(pred, target, voxelspacing=1, connectivity=18)

    if residual_mean_square_distance.shape == (0,):
        return np.inf
    # 在Mean Surface Distance上平方
    rms_surface_distance = (residual_mean_square_distance ** 2).mean()

    contrary_residual_mean_square_distance = metric.binary.__surface_distances(pred, target, voxelspacing=1, connectivity=18)
    if contrary_residual_mean_square_distance.shape == (0,):
        return np.inf

    # 在Mean Surface Distance上平方
    contrary_rms_surface_distance = (contrary_residual_mean_square_distance ** 2).mean()
    # 最后开方求均方根
    rms_distance = np.sqrt(np.mean((rms_surface_distance, contrary_rms_surface_distance)))
    return rms_distance


def main():
    import SimpleITK as itk
    print("************测试******************")
    itk_Mask = itk.ReadImage(r'D:\CCCFFF/ADNI/Mask.nii.gz')
    Mask = itk.GetArrayFromImage(itk_Mask)
    itk_Pre = itk.ReadImage(r'D:\CCCFFF\ADNI/pre.nii.gz')
    Pre = itk.GetArrayFromImage(itk_Pre)

    print(dice(Mask, Pre))
    print(jaccard(Mask, Pre))
    print(precision(Mask, Pre))
    print(recall(Mask, Pre))
    print(MeanDistance(Mask, Pre))  #
    print(Hausdorff(Mask, Pre))
    print(Hausdorff95(Mask, Pre))
    print(ASSD(Mask, Pre))  #
    print(RMSD(Mask, Pre))   #


if __name__ == '__main__':
    main()