from dataset.test_dataset import Mini_DataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
import torch.nn.functional as F
from utils_T import metrics, common
import SimpleITK as sitk
import os
import numpy as np
from model.Unet import UNet, RecombinationBlock


def test(model, dataset, save_path, test_data_path):
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    avg_dice1 = 0
    for idx, (data, target, data_GM) in tqdm(enumerate(dataloader), total=len(dataloader)):
        target = common.to_one_hot_3d(target.long())

        data, target, data_GM = data.float(), target.float(), data_GM.float()
        data, target, data_GM = data.to(device), target.to(device), data_GM.to(device)

        pro, pred = model(data, data_GM)

        val_loss = metrics.DiceMeanLoss()(pred, target)
        # val_loss = metrics.WeightDiceLoss()(output, target)
        val_dice0 = metrics.dice(pred, target, 0)
        val_dice1 = metrics.dice(pred, target, 1)
        # val_dice2 = metrics.dice(pred, target, 2)
        avg_dice1 += float(metrics.dice(pred, target, 1))

        pred_img = torch.argmax(pred, dim=1)
        pred_img = F.pad(pred_img, [0, 0, -2, -2, -2, -2])
        pred_img = pred_img.detach().cpu()

        img = sitk.GetImageFromArray(np.squeeze(np.array(pred_img.numpy(), dtype='uint8'), axis=0))
        nda = sitk.ReadImage('/home/ytusd3/lb/fin_HPC/100GM_And_R_to_L/train/ADNI_002_S_0295/cutImg.nii.gz')
        img.SetOrigin(nda.GetOrigin())
        img.SetDirection(nda.GetDirection())
        img.SetSpacing(nda.GetSpacing())

        filenameList = sorted(os.listdir(test_data_path))
        #filepath = test_data_path + '/' + filenameList[idx]
        filename = filenameList[idx]
        print(filename)
        if not os.path.exists(save_path + '/' + filename):os.mkdir(save_path + '/' + filename)
        pername = 'preMask.nii.gz'

        #sitk.WriteImage(img, os.path.join(save_path + '/' + filenameList[idx] + '/', pername))

        print('\nval_loss: {:.4f}\tdice0: {:.4f}\tdice1: {:.4f}\t\n'.format(
                val_loss, val_dice0, val_dice1))
        # return val_loss, val_dice0, val_dice1, val_dice2
        # return val_loss, val_dice0, val_dice1
    avg_dice1 /= len(dataloader)
    print(avg_dice1)


if __name__ == '__main__':
    args = config.args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    torch.cuda.set_device(0)

    # model info
    model = UNet(1, [32, 48, 64, 96, 128], 2, conv_block=RecombinationBlock, net_mode='3d').to(device)
    ckpt = torch.load('./output/{}/best_model10.pth'.format(args.save),map_location='cpu')
    # ckpt = torch.load('./output/{}/latest_model.pth'.format(args.save))
    model.load_state_dict(ckpt['net'])

    # data info
    test_data_path = r'/home/ytusd3/lb/fin_HPC/35GM_And_R_to_L/'
    result_save_path = r'/home/ytusd3/lb/resultNoAttens_HPC/'
    # result_save_path = r'E:/Datesets/last_model_result/'
    if not os.path.exists(result_save_path): os.mkdir(result_save_path)
    datasets = Mini_DataSet(test_data_path, resize_scale=args.resize_scale)

    test(model, datasets, result_save_path, test_data_path)
