import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# data in/out and dataset
parser.add_argument('--dataset_path_train', default='E:\Datesets\data35/100GM_And_R_to_L/train/', help='fixed trainset root path')
parser.add_argument('--dataset_path_val', default='E:\Datesets\data35/100GM_And_R_to_L/val/', help='fixed trainset root path')

parser.add_argument('--save', default='model1', help='save path of trained model')

parser.add_argument('--resize_scale', type=float, default=1, help='resize scale for input data')


parser.add_argument('--batch_size', type=list, default=2, help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0001)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 10)')



args = parser.parse_args()


