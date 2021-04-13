import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config:
    log = "./log"  # Path to save log
    checkpoint_path = "./checkpoints/"  # Path to store model
    resume = "./checkpoints/"
    evaluate = None  # 测试模型，evaluate为模型地址
    dataset_path = './datasets'
    train_dataset_path = dataset_path + '/CIFAR100'
    test_dataset_path = dataset_path + '/CIFAR100'
    device = 'cuda:0'

    split_nums = 30

    pretrained = False
    seed = 2021
    num_classes = 100

    milestones = [60, 120, 160]
    epochs = 120
    batch_size = 128
    accumulation_steps = 1
    lr = 0.1
    gamma = 0.2
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 4
    print_interval = 30
    apex = True

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_dataset_init = {
        "root": train_dataset_path,
        "train": True,
        "download": True,
        "transform": train_transform
    }
    test_dataset_init = {
        "root": test_dataset_path,
        "train": False,
        "download": True,
        "transform": test_transform
    }