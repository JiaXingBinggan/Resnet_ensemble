import os
import logging
from logging.handlers import TimedRotatingFileHandler

import numpy as np
import random

import torch
from PIL import Image
import pickle
import torch.utils.data.dataset as Dataset
from typing import Any, Callable, Optional, Tuple

def get_ensemble_logger(name, log_dir='log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ensemble_name = os.path.join(log_dir, '{}.ensemble.log'.format(name))
    ensemble_handler = TimedRotatingFileHandler(ensemble_name,
                                            when='D',
                                            encoding='utf-8')
    ensemble_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ensemble_handler.setFormatter(formatter)

    logger.addHandler(ensemble_handler)

    return logger


def get_logger(name, log_dir='log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    error_handler = TimedRotatingFileHandler(error_name,
                                             when='D',
                                             encoding='utf-8')
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

def ensemble_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res, pred


def load_train_data(data_root):
    data, targets = [], []
    for file_name in os.listdir(data_root + '/CIFAR100/cifar-100-python'):
        if file_name == 'train':
            file_path = os.path.join(data_root + '/CIFAR100/cifar-100-python', file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

    return np.vstack(data), np.vstack(targets)

def load_test_data(data_root):
    data, targets = [], []
    for file_name in os.listdir(data_root + '/CIFAR100/cifar-100-python'):
        if file_name == 'test':
            file_path = os.path.join(data_root + '/CIFAR100/cifar-100-python', file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

    return np.vstack(data), np.vstack(targets)


class CIFAR_SPLIT(Dataset.Dataset):
    def __init__(self,
                 data,
                 targets,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super(CIFAR_SPLIT, self).__init__()
        # ???????????????????????????????????????
        self.transform = transform
        self.target_transform = target_transform

        self.data = data
        self.targets = targets
        self.data = self.data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))  # convert to HWC
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    # ?????????????????????
    def __len__(self) -> int:
        return len(self.data)

    # ???????????????????????????
    def __getitem__(self, index) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

'''
1. ??????????????????????????????????????????????????????????????????????????????m?????????????????????D?????????m??????
???????????????D'??????????????????D?????????36.8%?????????????????????D????????????????????????D\D??????????????????
2. ????????????????????????????????????????????????/?????????????????????
3. ????????????????????????????????????????????????????????????????????????????????????????????????????????????
4. ??????????????????????????????????????????????????????????????????????????????
'''
class BootStrap(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sampling(self):
        _slice = []
        while len(_slice) < self.n_samples:
        #_slice = np.random.choice(self.n_samples, self.n_samples, replace=True)
            p = random.sample(range(self.n_samples // 2), 1)[0]

            _slice.append(p)
        return _slice