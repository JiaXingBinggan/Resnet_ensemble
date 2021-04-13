import numpy as np
import os
from PIL import Image
import torchvision
import pickle
import torchvision.transforms as transforms
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
from typing import Any, Callable, Optional, Tuple

class CIFAR_TRAIN(Dataset.Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super(CIFAR_TRAIN, self).__init__()
        # 初始化，定义数据内容和标签
        self.root = root
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        self._load_data()

    def _load_data(self):
        for file_name in os.listdir(self.root + '/cifar-100-python'):
            file_path = os.path.join(self.root + '/cifar-100-python', file_name)
            if file_name == 'file.txt~' or file_name == 'meta' or file_name == 'test':
                continue
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        print(self.data[np.random.choice(50000, 10000, replace=True), :].shape)


    # 返回数据集大小
    def __len__(self) -> int:
        return len(self.data)

    # 得到数据内容和标签
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

data = []
targets = []

train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

train_dataset_init = {
    "root": './data',
    "transform": train_transform
}

train_dataset = CIFAR_TRAIN(**train_dataset_init)
train_loader = DataLoader(train_dataset,
                          batch_size=128)

# for (img, label) in train_loader:
#     print(img, label)

