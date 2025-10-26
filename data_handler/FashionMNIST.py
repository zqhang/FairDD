import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets.mnist import FashionMNIST

class BiasedFashionMNIST(FashionMNIST):
    COLOUR_MAP = [[255, 0, 0], [255, 128, 0], [255, 255, 0], [255, 255, 128], [0, 255, 0], [0, 255, 255], [0, 0, 255],
                  [128, 0, 255], [255, 0, 255], [255, 255, 255]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=0.8, n_confusing_labels=9,is_foreground=True):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.is_foreground = is_foreground

        pad_images = torch.zeros(
            (self.data.shape[0], self.data.shape[1] + 4, self.data.shape[2] + 4),
            dtype=self.data.dtype,
        )

        # 将 self.images 的数据复制到 pad_images 的相应位置
        pad_images[:, 4: 4 + self.data.shape[1], 4: 4 + self.data.shape[2]] = self.data
        self.data=pad_images


        self.data, self.targets, self.biased_targets = self.build_biased_fashionmnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_fashionmnist(self, indices, label, is_foreground):
            raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_fashionmnist(self):

        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_fashionmnist(indices, bias_label, self.is_foreground)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)

        # colors
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, int(self.biased_targets[index])


class ColourBiasedFashionMNIST(BiasedFashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=0.8, n_confusing_labels=9,is_foreground=True):
        super(ColourBiasedFashionMNIST, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels,
                                                is_foreground=is_foreground)
    # 前景色，内容白
    def _binary_to_colour_foreground(self, data, colour):
        # 将数据归一化为 [0, 1] 范围
        data = data.float() / 255.0

        # 设置起始颜色为黑色
        start_colour = torch.tensor([0, 0, 0], dtype=torch.float32)

        # 将 colour 转换为张量，并归一化到 [0, 1] 范围
        end_colour = torch.tensor(colour, dtype=torch.float32) / 255.0

        # 扩展维度以便与数据相乘
        start_colour = start_colour.view(1, 1, 1, 3)
        end_colour = end_colour.view(1, 1, 1, 3)

        # 计算颜色梯度
        fg_data = start_colour + data.unsqueeze(3) * (end_colour - start_colour)

        # 将颜色梯度从 [0, 1] 转换回 [0, 255]
        fg_data = (fg_data * 255).byte()

        return fg_data

    # 背景色，内容黑
    def _binary_to_colour_background(self, data, colour):
        # 将数据归一化为 [0, 1] 范围
        data = (255.0-data.float()) / 255.0

        # 设置起始颜色为黑色
        # start_colour = torch.tensor([0, 0, 0], dtype=torch.float32)
        start_colour = torch.tensor([255.0, 255.0, 255.0], dtype=torch.float32)

        # 将 colour 转换为张量，并归一化到 [0, 1] 范围
        end_colour = torch.tensor(colour, dtype=torch.float32) / 255.0

        # 扩展维度以便与数据相乘
        start_colour = start_colour.view(1, 1, 1, 3)
        end_colour = end_colour.view(1, 1, 1, 3)

        # 计算颜色梯度
        fg_data = start_colour + data.unsqueeze(3) * (end_colour - start_colour)

        # 将颜色梯度从 [0, 1] 转换回 [0, 255]
        fg_data = (fg_data * 255).byte()

        return fg_data

    def _make_biased_fashionmnist(self, indices, label,is_foreground):
        if self.is_foreground:
            return self._binary_to_colour_foreground(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]
        else:
            return self._binary_to_colour_background(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]


def get_biased_fashionmnist_dataloader(root, batch_size, data_label_correlation,is_foreground,
                                n_confusing_labels=9, train=True, num_workers=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.3817, 0.1961, 0.2846),
    #                          std=(0.5573, 0.6763, 0.6183))])
    dataset = ColourBiasedFashionMNIST(root+"/FashionMNIST", train=train, transform=transform,
                                download=True, data_label_correlation=data_label_correlation,
                                n_confusing_labels=n_confusing_labels,
                                is_foreground=is_foreground)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataset,dataloader

