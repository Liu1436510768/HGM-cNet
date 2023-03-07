import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils_T.common import *


class Lits_DataSet(Dataset):
    def __init__(self, resize_scale, dataset_path, mode=None):
        self.resize_scale = resize_scale
        self.dataset_path = dataset_path
        self.n_labels = 2

        # 将数据集分成train和val集 （4: 1）
        # data_name_list = os.listdir(dataset_path + "/" + "data")
        # data_name_list = os.listdir(dataset_path)
        # data_num = len(data_name_list)
        # print('the fixed dataset total numbers of samples is :', data_num)
        # random.shuffle(data_name_list)

        # train_rate = 0.8
        # val_rate = 0.2
        #
        # assert val_rate + train_rate == 1.0
        # train_name_list = data_name_list[:int(data_num * train_rate)]
        # val_name_list = data_name_list[int(data_num * train_rate):int(data_num * (train_rate + val_rate))]

        if mode == 'train':
            data_name_list = os.listdir(dataset_path)
            data_num = len(data_name_list)
            print('the fixed dataset total numbers of samples is :', data_num)
            random.shuffle(data_name_list)

            self.filename_list = data_name_list
        elif mode == 'val':
            data_name_list = os.listdir(dataset_path)
            data_num = len(data_name_list)
            print('the fixed dataset total numbers of samples is :', data_num)
            random.shuffle(data_name_list)

            self.filename_list = data_name_list
        else:
            raise TypeError('Dataset mode error!!! ')

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
        return len(self.filename_list)

    def get_train_batch_by_index(self, index, resize_scale=1):

        img, label, data_GM = self.get_np_data_3d(self.filename_list[index], resize_scale=resize_scale)
        print(img.shape, label.shape,)

        img = img[np.newaxis, ...]
        data_GM = data_GM[np.newaxis, ...]

        #data_np = np.concatenate([img, data_GM], axis=0)
        data_np = img

        return data_np, label, data_GM

    # 读取文件数据返回图像np数组
    def get_np_data_3d(self, filename, resize_scale=1):

        data_np = sitk_read_raw(self.dataset_path + filename + '/cutImg.nii.gz',
                                resize_scale=resize_scale)
        data_np = norm_img(data_np)    # 归一化像素值到（0，1）之间

        print(self.dataset_path + filename + '/cutMask.nii.gz')

        data_GM = sitk_read_raw(self.dataset_path + filename + '/cutGM.nii.gz',
                                resize_scale=resize_scale)

        label_np = sitk_read_raw(self.dataset_path + filename + '/cutMask.nii.gz',
                                 resize_scale=resize_scale)

        return data_np, label_np, data_GM


# 测试代码
def main():
    fixd_path = '/media/zhyl/Files/libin/60R_TO_L_AND_L35_Cut/'
    dataset = Lits_DataSet(1, fixd_path, mode='train')  # batch size
    data_loader = DataLoader(dataset=dataset, batch_size=7, shuffle=True, num_workers=2)
    for data, data60, mask, mask60 in data_loader:

        print(data.shape, mask.shape)


if __name__ == '__main__':
    main()
