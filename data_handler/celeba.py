import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union, TypeVar, Iterable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

T = TypeVar("T", str, bytes)
CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebA_train(Dataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (Tensor shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (Tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (Tensor shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
            self,
            target_label_idx: int,
            sensitive_label_idx: int,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(CelebA_train, self).__init__()
        self.target_label_idx = target_label_idx
        self.sensitive_label_idx = sensitive_label_idx
        self.root = root
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.transform = transform
        self.target_transform = target_transform
        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        # self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.floor_divide(self.attr + 1, 2)
        self.attr_names = attr.header

    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []
        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        # return X,target[2],target[20] # 有无吸引力/性别
        # return X,target[2],target[20] # 有无吸引力/性别
        return X, target[self.target_label_idx], target[self.sensitive_label_idx]

    def __len__(self) -> int:
        return len(self.attr)


import numpy as np
import random


class CelebA_test(Dataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (Tensor shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (Tensor shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (Tensor shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
            self,
            target_label_idx: int,
            sensitive_label_idx: int,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(CelebA_test, self).__init__()
        self.target_label_idx = target_label_idx
        self.sensitive_label_idx = sensitive_label_idx
        self.root = root
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.transform = transform
        self.target_transform = target_transform
        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split]
        splits = self._load_csv("list_eval_partition.txt")

        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

        # self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # self.identity = identity[mask]

        # map from {-1, 1} to {0, 1}
        self.attr = torch.floor_divide(self.attr + 1, 2)
        self.attr_names = attr.header

        lab_idx_dict = {}
        col_idx_dict = {}

        lab = self.attr[:, self.target_label_idx]
        col = self.attr[:, self.sensitive_label_idx]

        for lab_id in np.unique(lab):
            lab_idx_dict[lab_id] = [idx for idx, c in enumerate(lab) if lab_id == c]
        for col_id in np.unique(col):
            col_idx_dict[col_id] = [idx for idx, c in enumerate(col) if col_id == c]

        test_idx = []
        min_intersection = 1e10
        for i in range(len(lab_idx_dict)):
            for j in range(len(col_idx_dict)):
                intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
                min_intersection = min(min_intersection, len(intersection))

        for i in range(len(lab_idx_dict)):
            for j in range(len(col_idx_dict)):
                intersection = list(set(lab_idx_dict[i]) & set(col_idx_dict[j]))
                min_intersection = min(min_intersection, len(intersection))
                # print(i, j, min_intersection, len(intersection))
                select_idx = random.sample(intersection, min(len(intersection), min_intersection))
                test_idx.extend(select_idx)
                lab_idx_dict[i] = list(set(lab_idx_dict[i]) - set(select_idx))
                col_idx_dict[j] = list(set(col_idx_dict[j]) - set(select_idx))

        self.attr = self.attr[test_idx]
        temp = []
        for i in test_idx:
            temp.append(self.filename[i])
        self.filename = temp
        self.bbox = self.bbox[test_idx]
        self.landmarks_align = self.landmarks_align[test_idx]
        # self.identity = self.identity[test_idx]

    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []
        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        # return X, target[2], target[20]  # 有无吸引力/性别
        # return X,target[31],target[20] # 微笑/性别
        return X, target[self.target_label_idx], target[self.sensitive_label_idx]

    def __len__(self) -> int:
        return len(self.attr)


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


def CelebA(target_label_idx, sensitive_label_idx, data_dir="/mnt/DatasetCondensation-master/data/celeba"):
    mean = (0.5063, 0.4258, 0.3832)
    std = (0.2676, 0.2453, 0.2410)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), ])
    train_dataset = CelebA_train(target_label_idx, sensitive_label_idx, root=data_dir, split='train',
                                 transform=transform)
    test_dataset = CelebA_test(target_label_idx, sensitive_label_idx, root=data_dir, split='test', transform=transform)
    return train_dataset, test_dataset, mean, std


if __name__ == "__main__":
    # target_label_idx = 31  # 微笑
    # target_label_idx = 39  # 年轻
    target_label_idx = 33  # 卷发
    # target_label_idx=2 # 有无吸引力
    sensitive_label_idx = 20  # 性别

    for i in range(0,40):

        target_label_idx=i
        print(target_label_idx)
        train_dataset, test_dataset, mean, std = CelebA(target_label_idx, sensitive_label_idx,
                                                        data_dir="/mnt/DatasetCondensation-master/data/celeba")
        # dataloader = DataLoader(train_dataset, batch_size=32)
        # for x, tra, m in dataloader:
        #     print(x.shape)
        #     print(tra, m)

        train_target=train_dataset.attr[:, target_label_idx]
        train_sensitive=train_dataset.attr[:, sensitive_label_idx]

        test_target=test_dataset.attr[:, target_label_idx]
        test_sensitive=test_dataset.attr[:, sensitive_label_idx]

        aa=0
        ab=0
        ba=0
        bb=0

        for index in range(len(train_target)):
            target=train_target[index]
            sensitive=train_sensitive[index]
            if target==0 and sensitive==0:
                aa=aa+1
            if target==0 and sensitive==1:
                ab=ab+1
            if target==1 and sensitive==0:
                ba=ba+1
            if target==1 and sensitive==1:
                bb=bb+1

        print((aa,ab,ba,bb))

        aa = 0
        ab = 0
        ba = 0
        bb = 0

        for index in range(len(test_target)):
            target=test_target[index]
            sensitive=test_sensitive[index]
            if target==0 and sensitive==0:
                aa=aa+1
            if target==0 and sensitive==1:
                ab=ab+1
            if target==1 and sensitive==0:
                ba=ba+1
            if target==1 and sensitive==1:
                bb=bb+1

        print((aa,ab,ba,bb))
        print("--------------------------")

    # L=[]
    # LL=[]
    # for i in range(10):
    #     L.append(train_dataset[i][1])
    #     LL.append(train_dataset[i][2])
    # print(L)
    # print(LL)

    # from torchvision.utils import save_image
    #
    # ''' visualize and save '''
    # save_name = '12345.png'
    # image_syn_vis = train_dataset[1][0]
    # print(train_dataset[1][1])
    # print(train_dataset[1][2])
    # mean = (0.5063, 0.4258, 0.3832)
    # std = (0.2676, 0.2453, 0.2410)
    # for ch in range(3):
    #     image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
    # image_syn_vis[image_syn_vis < 0] = 0.0
    # image_syn_vis[image_syn_vis > 1] = 1.0
    # save_image(image_syn_vis, save_name,
    #            nrow=100)  # Trying normalize = True/False may get better visual effects.
