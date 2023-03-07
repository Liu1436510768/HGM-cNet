from collections import OrderedDict

import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import dataset_list
from model.Unet import UNet, RecombinationBlock

from utils_T import logger, init_util, metrics, common
from torch.cuda.amp import autocast as autocast, GradScaler


def val(model, val_loader):     # 读取val数据测试模型
    model.eval()   # 不启用 BatchNormalization 和 Dropout。将BatchNormalization和Dropout置为False
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    with torch.no_grad():                 # 枚举
        for idx, (data, target, GM) in tqdm(enumerate(val_loader), total=len(val_loader)):
            target = common.to_one_hot_3d(target.long())

            # data = F.pad(data, pad=(1, 1, 1, 1, 0, 0), mode="constant", value=0)  # 2,1,24,32,32
            # target = F.pad(target, pad=(1, 1, 1, 1, 0, 0), mode="constant", value=0)

            data, target, GM = data.float(), target.float(), GM.float()
            data, target, GM = data.to(device), target.to(device), GM.to(device)
            #output = model(data, data_GM)
            prob, output = model(data, GM)

            loss = metrics.DiceMeanLoss()(output, target)



            dice0 = metrics.dice(output, target, 0)
            dice1 = metrics.dice(output, target, 1)

            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)

    val_loss /= len(val_loader)
    val_dice0 /= len(val_loader)
    val_dice1 /= len(val_loader)

    return OrderedDict({'Val Loss': val_loss, 'Val dice0': val_dice0,
                        'Val dice1': val_dice1})


def train(model, train_loader):
    print("=======Epoch:{}=======".format(epoch))
    model.train()   # 启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    for idx, (data, target, GM) in tqdm(enumerate(train_loader), total=len(train_loader)):
        target = common.to_one_hot_3d(target.long())

        # data = F.pad(data, pad=(1, 1, 1, 1, 0, 0), mode="constant", value=0)    # 2,1,48,60,60
        # target = F.pad(target, pad=(1, 1, 1, 1, 0, 0), mode="constant", value=0)
        data, target, GM = data.float(), target.float(), GM.float()
        data, target, GM = data.to(device), target.to(device), GM.to(device)
        # output = model(data, data_GM)
        optimizer.zero_grad()  # 把梯度置零
        with autocast():
            prob, output = model(data, GM)



        # loss = nn.CrossEntropyLoss()(output,target)
        # loss=metrics.SoftDiceLoss()(output,target)
        # loss=nn.MSELoss()(output,target)
            loss1 = metrics.DiceMeanLoss()(prob, target)
            loss2 = metrics.DiceMeanLoss()(output, target)
        # loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.CrossEntropy()(output,target)
            loss = loss2 + loss1
        scaler.scale(loss).backward()
        #optimizer.step()   # 更新参数
        scaler.step(optimizer)
        scaler.update()

        train_loss += float(loss)
        train_dice0 += float(metrics.dice(output, target, 0))
        train_dice1 += float(metrics.dice(output, target, 1))

    train_loss /= len(train_loader)
    train_dice0 /= len(train_loader)
    train_dice1 /= len(train_loader)

    return OrderedDict({'Train Loss': train_loss, 'Train dice0': train_dice0,
                        'Train dice1': train_dice1})


if __name__ == '__main__':
    args = config.args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    torch.cuda.set_device(0)
    # # data info
    train_set = dataset_list.Lits_DataSet(args.resize_scale,
                                          args.dataset_path_train, mode='train')
    val_set = dataset_list.Lits_DataSet(args.resize_scale,
                                        args.dataset_path_val, mode='val')
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True)
    # model info
    model = UNet(1, [32, 48, 64, 96, 128], 2, conv_block=RecombinationBlock, net_mode='3d').to(device)

    # ckpt = torch.load('./output/{}/best_model4.pth'.format(args.save))
    # model.load_state_dict(ckpt['net'])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    init_util.print_network(model)

    log = logger.Logger('./output/{}'.format(args.save))

    best = [0, np.inf]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器

    scaler = GradScaler()

    for epoch in range(1, args.epochs + 1):
        # data info
        # train_set = dataset_list.Lits_DataSet(args.resize_scale,
        #                                       args.dataset_path_train, mode='train')
        # val_set = dataset_list.Lits_DataSet(args.resize_scale,
        #                                     args.dataset_path_val, mode='val')
        # train_loader = DataLoader(dataset=train_set, shuffle=True)
        # val_loader = DataLoader(dataset=val_set, shuffle=True)

        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader)
        val_log = val(model, val_loader)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # torch.save(state, os.path.join('./output/{}'.format(args.save), 'latest_model.pth'))
        #trigger += 1
        if val_log['Val Loss'] < best[1]:
            print('Saving best model')
            torch.save(state, os.path.join('./output/{}'.format(args.save), 'best_model10.pth'))
            best[0] = epoch
            best[1] = val_log['Val Loss']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
