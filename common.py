import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch
import os
import random


# 读取file_path文件下的数据 返回列表
def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list


# 创建索引txt文件方法 将数据集分成train和val集 （4:1）
def write_train_val_test_name_list(fixd_path):
    data_name_list = os.listdir(fixd_path + "/" + "data")
    data_num = len(data_name_list)
    print('the fixed dataset total numbers of samples is :', data_num)
    random.shuffle(data_name_list)

    train_rate = 0.8
    val_rate = 0.2

    assert val_rate + train_rate == 1.0
    train_name_list = data_name_list[0:int(data_num * train_rate)]
    val_name_list = data_name_list[int(data_num * train_rate):int(data_num * (train_rate + val_rate))]

    write_name_list(fixd_path, train_name_list, "train_name_list.txt")
    write_name_list(fixd_path, val_name_list, "val_name_list.txt")


# 输出txt索引文件
def write_name_list(fixd_path, name_list, file_name):
    f = open(fixd_path + file_name, 'w')
    for i in range(len(name_list)):
        f.write(str(name_list[i]) + "\n")
    f.close()


# 读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
def sitk_read_raw(img_path, resize_scale=1):
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError("input img is None!!!")
    # image转成array，且从[depth, width, height]转成[width, height, depth]
    nda = sitk.GetArrayFromImage(nda)  # channel first
    nda = ndimage.zoom(nda, [resize_scale, resize_scale, resize_scale], order=0)  # rescale

    return nda


# target one-hot编码
# def to_one_hot_3d(tensor, n_classes=2):  # shape = [batch, s, h, w]
def to_one_hot_3d(tensor, n_classes=2):
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot


# 对输入的volume数据x，对每个像素值进行one-hot编码
def make_one_hot_3d(x, n):
    one_hot = np.zeros([x.shape[0], x.shape[1], x.shape[2], n])  # 创建one-hot编码后shape的zero张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                one_hot[i, j, v, int(x[i, j, v])] = 1  # 给相应类别的位置置位1，模型预测结果也应该是这个shape
    return one_hot


MIN_BOUND = 0
MAX_BOUND = 255


# 归一化像素值到（0，1）之间
def norm_img(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image[image > 1] = 1.
    # image[image < 0] = 0.
    return image


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 100:
       lr = args.lr * (0.1 ** (epoch // 30))
    else:
       lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr