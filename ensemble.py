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
from utils import get_ensemble_logger, AverageMeter, ensemble_accuracy

import numpy as np


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
                        default=Config.batch_size,
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

def test(test_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        final_preds = torch.LongTensor().to(Config.device)
        for inputs, labels in test_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            acc1, preds = ensemble_accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0].item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            final_preds = torch.cat([final_preds, preds], dim=-1)

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, throughput, final_preds.view(-1, 1)


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
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True)
    model = resnet34(**{
        "pretrained": args.pretrained,
        "num_classes": args.num_classes,
    })

    for name, param in model.named_parameters():
        logger.info(f"{name},{param.requires_grad}")

    model = model.to(Config.device)

    model = nn.DataParallel(model)

    bootstrap_preds = torch.LongTensor().to(Config.device)
    for boostrap_iter in range(1, 3):
        if not os.path.isfile(args.resume + "best" + str(boostrap_iter) + ".pth"):
            raise Exception(
                f"{args.resume + 'best.' + str(boostrap_iter) + '.pth'} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.resume + 'best' + str(boostrap_iter) + '.pth'}")
        checkpoint = torch.load(args.resume + 'best' + str(boostrap_iter) + '.pth',
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        best_test_acc1, throughput, best_preds = test(test_loader, model, args)
        logger.info(
            f"Test: boostrap_iter {checkpoint['boostrap_iter']:0>3d}, "
            f"top1 acc: {best_test_acc1:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        logger.info(f"finish training, boostrap_iter {boostrap_iter:0>3d}, best test acc: {best_test_acc1:.2f}%")

        if not os.path.exists(args.record):
            os.makedirs(args.record)

        boostrap_iter_record = pd.DataFrame(data=best_preds.cpu().numpy(), columns=['test_labels'])
        boostrap_iter_record.to_csv(args.record + '/bootstrap_iter_' + str(boostrap_iter) + '.csv')

        bootstrap_preds = torch.cat([bootstrap_preds, best_preds], dim=-1)
    print(bootstrap_preds)