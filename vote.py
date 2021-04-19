import sys
import os
import argparse
import random
import time
import warnings
import pandas as pd

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
from scipy import stats


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

    bootstrap_test_acc1_dices = pd.read_csv(args.record +
                                            '/bootstrap_iter_test_acc1.csv')
    max_acc1 = max(bootstrap_test_acc1_dices.values[0])
    print('max acc1', max_acc1)

    boostrap_iter_test_labels = pd.read_csv(args.record + '/bootstrap_iter_test_labels.csv')

    boostrap_iter_test_labels.mode(axis=1).to_csv(args.record + '/test.csv', index=None)
    origin_vote_preds = np.array(list(map(lambda x: sorted(x)[len(x) // 2],
                                                    boostrap_iter_test_labels.values)))

    vote_preds = torch.LongTensor(origin_vote_preds.reshape(-1, 1))
    vote_targets = torch.LongTensor(test_target_np)
    acc1, preds = accuracy(vote_preds, vote_targets, topk=(1,))
    print('vote acc1', acc1[0].item())

    vote_pred_df = pd.DataFrame(data=origin_vote_preds.reshape(-1, 1))
    vote_pred_df.to_csv(args.record + '/vote_preds.csv', index=None)

