import sys
import os
import argparse
import random
import time
import warnings
import pandas as pd

from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from apex import amp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from thop import profile
from thop import clever_format
from torch.utils.data import DataLoader
from config import Config
from resnetforcifar import resnet34
from torchvision.datasets import CIFAR100
from utils import get_ensemble_logger, AverageMeter, ensemble_accuracy, load_test_data

import numpy as np
from scipy.spatial.distance import pdist


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=Config.momentum,
                        help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--gamma',
                        type=float,
                        default=Config.gamma,
                        help='gamma')
    parser.add_argument('--milestones',
                        type=list,
                        default=Config.milestones,
                        help='optimizer milestones')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='batch size')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=Config.accumulation_steps,
                        help='gradient accumulation steps')
    parser.add_argument('--fine_tuning',
                        type=bool,
                        default=Config.fine_tuning,
                        help='use fine-tying model')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='load pretrained model params or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--record',
                        type=str,
                        default=Config.record,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')

    return parser.parse_args()


def accuracy(pred, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        pred = pred.t().view(1, -1)

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res, pred


def pearson_diff(pearsonr_n, select_arrays, max_acc1_index, max_acc1_preds, boostrap_iter_test_labels):
    pearsonr_x_ys = []
    for i in select_arrays:
        pearsonr_x_ys.append([i, pearsonr(max_acc1_preds, boostrap_iter_test_labels.iloc[:, i])[0]])
    sort_pearsonr_x_ys = sorted(pearsonr_x_ys, key=lambda s: s[1], reverse=True)

    diff_indexs = np.array(sort_pearsonr_x_ys[:pearsonr_n])

    return list(diff_indexs[:, 0].astype(int)) + [max_acc1_index]


if __name__ == '__main__':
    args = parse_args()
    logger = get_ensemble_logger(__name__, args.log)

    if not torch.cuda.is_available():
        print("suggest to use gpu to train network!")
        Config.device = 'cpu'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True

    # dataset and dataloader
    test_dataset = CIFAR100(**Config.test_dataset_init)
    test_img_np, test_target_np = load_test_data(Config.dataset_path)

    bootstrap_test_acc1s = pd.read_csv(args.record +
                                            '/bootstrap_iter_test_acc1.csv').values[0]
    max_acc1 = max(bootstrap_test_acc1s)
    print('max acc1', max_acc1)
    max_acc1_index = list(bootstrap_test_acc1s).index(max_acc1) # from 0

    boostrap_iter_test_labels = pd.read_csv(args.record + '/bootstrap_iter_test_labels.csv')
    # max_acc1_preds = pd.read_csv(args.record + '/vote_preds.csv', dtype=int)
    max_acc1_preds = boostrap_iter_test_labels.iloc[:, max_acc1_index]


    # pearsonr_n = 5
    # origin_indexs = [j for j in range(Config.split_nums) if j != max_acc1_index] # except the index for max acc1
    # ensemble_lists = pearson_diff(pearsonr_n, origin_indexs, max_acc1_index, max_acc1_preds, boostrap_iter_test_labels)
    #
    # merges = boostrap_iter_test_labels.iloc[:, ensemble_lists]
    # vote_preds = torch.LongTensor(merges.mode(axis=1).values[:, 0])
    # vote_targets = torch.LongTensor(test_target_np)
    # acc1, preds = accuracy(vote_preds, vote_targets, topk=(1,))
    # print('ensemble acc1', acc1[0].item())
    #
    # current_acc1 = max_acc1
    # while current_acc1 >= max_acc1:
    top_n = 15
    top_n_indexs = list(np.argsort(-bootstrap_test_acc1s)[1: top_n + 1])
    # top_n_indexs = list(range(Config.split_nums))
    top_n_len = len(top_n_indexs)
    top_n_test_labels = boostrap_iter_test_labels.iloc[:, top_n_indexs]
    top_n_test_labels.columns = list(range(top_n_len))

    current_acc1 = max_acc1
    while current_acc1 >= max_acc1:
        max_acc1 = current_acc1
        pearsonr_x_ys = []
        for i in range(top_n_len):
            pearsonr_x_ys.append([i, pearsonr(max_acc1_preds.values.flatten(),
                                              top_n_test_labels.iloc[:, i].values)[0]])
            # pearsonr_x_ys.append([i, pdist([max_acc1_preds.values.flatten(),
            #                                               top_n_test_labels.iloc[:, i].values], 'cosine')])

        sort_pearsonr_x_ys = np.array(sorted(pearsonr_x_ys, key=lambda x: x[1])).astype(int)
        max_diff_index = list(sort_pearsonr_x_ys[:1][:, 0])

        merges = pd.concat([max_acc1_preds, top_n_test_labels.iloc[:, max_diff_index]], axis=1)
        vote_preds = torch.LongTensor(np.array(list(map(lambda x: sorted(x)[len(x) // 2], merges.values))).reshape(-1, 1))
        # vote_preds = torch.LongTensor(merges.mode(axis=1).values[:, 0])
        vote_targets = torch.LongTensor(test_target_np)
        acc1, preds = accuracy(vote_preds, vote_targets, topk=(1,))
        print('ensemble acc1', acc1[0].item())

        max_acc1_preds = pd.DataFrame(data=np.array(list(map(lambda x: sorted(x)[len(x) // 2], merges.values))))


        current_acc1 = acc1[0].item()
        top_n_test_labels = top_n_test_labels.drop(max_diff_index, axis=1)
        top_n_len -= len(max_diff_index)
        top_n_test_labels.columns = list(range(top_n_len))
