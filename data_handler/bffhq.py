import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from glob import glob

class bFFHQDataset(Dataset):
    target_attr_index = 0
    bias_attr_index = 1

    def __init__(self, root, split, transform=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.target_attrs = []
        self.bias_attrs = []

        if split == '0.5pct':
            self.align = glob(os.path.join(root, split, 'align', "*", "*"))
            self.conflict = glob(os.path.join(root, split, 'conflict', "*", "*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root, split, "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, split, "*"))

        # 提取属性标签并保存到成员变量中
        for fpath in self.data:
            first_attr = int(fpath.split('_')[-2])
            second_attr = int(fpath.split('_')[-1].split('.')[0])
            self.target_attrs.append(first_attr)
            self.bias_attrs.append(second_attr)

        self.target_attrs = torch.LongTensor(self.target_attrs)
        self.bias_attrs = torch.LongTensor(self.bias_attrs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        fpath = self.data[index]
        # first_attr = int(fpath.split('_')[-2])
        # second_attr = int(fpath.split('_')[-1].split('.')[0])
        # attr = torch.LongTensor([first_attr, second_attr])
        image = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        first_attr = self.target_attrs[index]
        second_attr = self.bias_attrs[index]

        # return image, attr
        return image, first_attr, second_attr # first_attr label,second_attr sensitive
        # return image, second_attr, first_attr # first_attr label,second_attr sensitive

def BFFHQ(data_dir):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), ])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = bFFHQDataset(root = data_dir, split = '0.5pct', transform=transform)
    test_dataset = bFFHQDataset(root = data_dir, split = 'test', transform=transform)
    return train_dataset, test_dataset, mean, std