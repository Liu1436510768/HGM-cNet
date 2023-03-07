from glob import glob

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils_T.common import *


class Mini_DataSet(Dataset):
    def __init__(self, dataset_path, resize_scale=1,):
        self.dataset_path = dataset_path
        self.resize_scale = resize_scale
        self.n_labels = 2

        self.data_list = sorted(os.listdir(dataset_path))
        print("The numbers of testset is ", len(self.data_list))

    def __getitem__(self, index):

        data, target, data_GM = self.get_train_batch_by_index(index=index, resize_scale=self.resize_scale)

        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        data_GM = torch.from_numpy(data_GM)
        data = F.pad(data, [0, 0, 2, 2, 2, 2], "constant", value=0)
        target = F.pad(target, [0, 0, 2, 2, 2, 2], "constant", value=0)
        data_GM = F.pad(data_GM, [0, 0, 2, 2, 2, 2], "constant", value=0)
        return data, target, data_GM

    def __len__(self):
        return len(self.data_list)

    def get_train_batch_by_index(self, index, resize_scale=1):

        img, label, data_GM = self.get_np_data_3d(self.data_list[index], resize_scale=resize_scale)
        print(img.shape, label.shape)

        img = img[np.newaxis, ...]
        data_GM = data_GM[np.newaxis, ...]

        return img, label, data_GM

    def get_np_data_3d(self, filename, resize_scale=1):
        data_np = sitk_read_raw(self.dataset_path + filename + '/cutImg.nii.gz',
                                resize_scale=resize_scale)

        data_np = norm_img(data_np)  # 归一化像素值到（0，1）之间

        print(self.dataset_path + filename + '/cutMask.nii.gz')

        data_GM = sitk_read_raw(self.dataset_path + filename + '/cutGM.nii.gz',
                               resize_scale=resize_scale)

        label_np = sitk_read_raw(self.dataset_path + filename + '/cutMask.nii.gz',
                             resize_scale=resize_scale)

        return data_np, label_np, data_GM


def main():
    test_path = r'E:/Datesets/data35/imageAtlas/'
    dataset = Mini_DataSet(test_path, 2)
    data_loader = DataLoader(dataset=dataset, num_workers=0, shuffle=False)
    print(len(data_loader))
    for data, mask in data_loader:
        data = torch.squeeze(data, dim=0)
        mask = torch.squeeze(mask, dim=0)
        print(data.shape, mask.shape)


if __name__ == '__main__':
    main()
