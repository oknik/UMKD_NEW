import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
from PIL import Image
import copy
import pandas as pd
import csv
import random
from typing import Tuple, Dict
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import pickle

class INDataset(data_utils.Dataset):

    def __init__(self, patch_level, img_root, dataset, transform=None, fold=3, balanced=False):
        self.data_list = []
        # self.items = []
        self.transform = transform
        self.patch_level = patch_level
        self.balanced = balanced
        # if dataset == 'train':
        #     data_num = [i for i in range(10) if i != fold]
        # elif dataset == 'valid':
        #     data_num = [fold]
#########训练和测试集转换为8：2##########
        if dataset == 'train' or dataset == 'valid':
            if dataset == 'train':
                data_num = [i for i in range(10) if i != fold and i != (fold + 1) % 10]
            elif dataset == 'valid':
                data_num = [fold, (fold + 1) % 10]
            print("Use the data fold: {}".format(data_num))
            for i in data_num:
                if self.balanced:
                    f = open('/root/autodl-tmp/UMKD_new/IN/split/ten_fold_balance/fold_{}.csv'.format(i), "r")
                else:
                    f = open('/root/autodl-tmp/UMKD_new/IN/split/ten_fold/fold_{}.csv'.format(i), "r")
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    self.data_list.append(row)
        elif dataset == 'test':
            test_csv = '/root/autodl-tmp/UMKD_new/IN/split/test.csv'
            with open(test_csv, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    self.data_list.append(row)

        self.label_list = [int(x[-1]) for x in self.data_list] #提出了所有折中的label
        # 提取出0, 1, 2
        #self.label_list = [x for x in self.label_list if x in {0, 1, 2}]

        if self.patch_level == 1:
            self.patch_class = self.get_token_class(fold)

        self.label_num = [0, 0, 0]
        for each in self.label_list:
            self.label_num[each] += 1
        print(self.label_num) #[23229, 2199, 4763, 786, 638]
        print(len(self.data_list)) #31615

    def __getitem__(self, idx):
        item = copy.deepcopy(self.data_list[idx])

        id_ = item[0]          # 现在这里是 id
        label = int(item[1])

        img_name_C = f"{id_}-{label}-C.png"
        img_name_G = f"{id_}-{label}-G.png"

        img_path_C = os.path.join('/root/autodl-tmp/UMKD_new/IN/images/', img_name_C)
        img_path_G = os.path.join('/root/autodl-tmp/UMKD_new/IN/images/', img_name_G)
        
        img_C = Image.open(img_path_C).convert('RGB')
        img_G = Image.open(img_path_G).convert('RGB')

        if self.transform:
            seed = np.random.randint(2147483647)

            random.seed(seed)
            torch.manual_seed(seed)
            img_C = self.transform(img_C)

            random.seed(seed)
            torch.manual_seed(seed)
            img_G = self.transform(img_G)

        img = torch.cat([img_C, img_G], dim=0)

        return img, label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    tran = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_data = SICAPv2Dataset(None,None,'train',tran,0)
    # loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    for i, (a, b) in enumerate(loader):
        print('i:',i)
        print('###',a.shape, b.shape)
        break