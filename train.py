import sys
import os
import argparse
import random
import time
import warnings

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
from utils import DataPrefetcher, get_logger, AverageMeter, accuracy, load_train_data, CIFAR_SPLIT, BootStrap

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


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger,
          args):
    top1 = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    iters = len(train_loader.dataset) // args.batch_size
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1

    while inputs is not None:
        inputs, labels = inputs.to(Config.device), labels.to(Config.device).squeeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / args.accumulation_steps

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if iter_index % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1 = accuracy(outputs, labels, topk=(1, ))[0]

        top1.update(acc1.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        inputs, labels = prefetcher.next()

        if iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, top1 acc: {acc1.item():.2f}%, loss_total: {loss.item():.2f}"
            )

        iter_index += 1

    scheduler.step()

    return top1.avg, losses.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(Config.device), labels.to(Config.device).squeeze(1)
            outputs = model(inputs)
            acc1 = accuracy(outputs, labels, topk=(1, ))[0]
            top1.update(acc1.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, throughput

def test(test_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in test_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            acc1 = accuracy(outputs, labels, topk=(1, ))[0]
            top1.update(acc1.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, throughput

def main(logger, args, train_loader, val_loader, test_loader, boostrap_iter):
    start_time = time.time()

    logger.info(f"creating model")
    model = resnet34(**{
        "pretrained": args.pretrained,
        "num_classes": args.num_classes,
    })

    for name, param in model.named_parameters():
        logger.info(f"{name},{param.requires_grad}")

    model = model.to(Config.device)
    criterion = nn.CrossEntropyLoss().to(Config.device)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    model = nn.DataParallel(model)

    best_val_acc = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume + "latest." + str(boostrap_iter) + ".pth"):
        logger.info(f"start resuming model from {args.resume + 'latest.' + str(boostrap_iter) + '.pth'}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {args.re0sume}, boostrap_iter {checkpoint['boostrap_iter']}, "
            f"epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, best_val_acc: {checkpoint['best_val_acc']:.2f}%, lr: {checkpoint['lr']:.6f}, "
            f"top1_acc: {checkpoint['acc1']}%")

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        acc1, losses = train(train_loader, model, criterion, optimizer,
                                   scheduler, epoch, logger, args)

        logger.info(
            f"train: boostrap_iter {boostrap_iter:0>3d}, epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, losses: {losses:.2f}"
        )

        acc1, throughput = validate(val_loader, model, args)
        logger.info(
            f"val: boostrap_iter {boostrap_iter:0>3d}, epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        if acc1 > best_val_acc:
            torch.save({
                           'boostrap_iter': boostrap_iter,
                           'loss': losses,
                           'model_state_dict': model.state_dict(),
                       },
                       os.path.join(args.checkpoints, "best" + str(boostrap_iter) + ".pth"))
            best_val_acc = acc1

        # remember best prec@1 and save checkpoint

        torch.save(
            {
                'boostrap_iter': boostrap_iter,
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'acc1': acc1,
                'loss': losses,
                'lr': scheduler.get_lr()[0],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.checkpoints, 'latest' + str(boostrap_iter) + '.pth'))

    if not os.path.isfile(args.resume + "best" + str(boostrap_iter) + ".pth"):
        raise Exception(
            f"{args.resume + 'best.' + str(boostrap_iter) + '.pth'} is not a file, please check it again")
    logger.info('start only evaluating')
    logger.info(f"start resuming model from {args.evaluate}")
    checkpoint = torch.load(args.resume + 'best' + str(boostrap_iter) + '.pth',
                            map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    best_test_acc1, throughput = test(test_loader, model, args)
    logger.info(
        f"Test: boostrap_iter {checkpoint['boostrap_iter']:0>3d}, "
        f"top1 acc: {best_test_acc1:.2f}%, throughput: {throughput:.2f}sample/s"
    )

    logger.info(f"finish training, boostrap_iter {boostrap_iter:0>3d}, best test acc: {best_test_acc1:.2f}%")
    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(__name__, args.log)

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

    no_split_train_dataset = CIFAR100(**Config.train_dataset_init) # 未做bootstrap采样的完整训练数据集，该行主要用作下载数据
    '''
    进行bootstrap数据集划分
    '''
    no_split_train_img_np, no_split_train_target_np = load_train_data(Config.dataset_path)

    boostrap = BootStrap(no_split_train_img_np.shape[0]) # 初始化Bootstrap采样器
    for boostrap_iter in range(30):
        train_slice = boostrap.sampling()
        val_slice = list(set(list(range(no_split_train_img_np.shape[0]))).difference(set(train_slice))) # 获取验证集下标

        split_train_img_np, split_train_target_np = \
            no_split_train_img_np[train_slice], no_split_train_target_np[train_slice] # 采样后的训练集

        split_val_img_np, split_val_target_np = \
            no_split_train_img_np[val_slice], no_split_train_target_np[val_slice]  # 采样后的验证集

        train_dataset_init = {
            "data": split_train_img_np,
            "targets": split_train_target_np,
            "transform": Config.train_transform
        }
        
        train_dataset = CIFAR_SPLIT(**train_dataset_init)
        train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

        val_dataset_init = {
            "data": split_val_img_np,
            "targets": split_val_target_np,
            "transform": Config.test_transform
        }

        val_dataset = CIFAR_SPLIT(**val_dataset_init)
        val_loader = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)

        main(logger, args, train_loader, val_loader, test_loader, boostrap_iter + 1)