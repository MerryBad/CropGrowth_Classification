""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image, decode_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=["file_Name", "pl_Name", "pl_Type", "pl_Step"], header=0)
        self.img_dir = img_dir
        self.transform = transform

        self.le_super = LabelEncoder()
        self.le_super = self.le_super.fit(self.img_labels['pl_Name'])
        self.le_super_class = self.le_super.classes_
        self.le_finer = LabelEncoder()
        self.le_finer = self.le_finer.fit(self.img_labels['pl_Step'])
        self.le_finer_class = self.le_finer.classes_

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # image = decode_image(image)
        super_Class = self.le_super.transform([self.img_labels.iloc[idx, 1]])
        finer_Class = self.le_finer.transform([self.img_labels.iloc[idx, 3]])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, super_Class, finer_Class  # , self.le_super_class, self.le_finer_class


# img_labels = pd.read_csv('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/label.csv',
#                          names=["file_Name", "pl_Name", "pl_Type", "pl_Step"], header=0)
# le_super = LabelEncoder()
# le_super = le_super.fit(img_labels['pl_Name'])
# le_super_class = le_super.classes_
# le_finer = LabelEncoder()
# le_finer = le_finer.fit(img_labels['pl_Step'])
# le_finer_class = le_finer.classes_
#
# print(le_super.classes_)
# print(le_finer_class)
# print(len(le_finer_class))

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from tqdm.notebook import tqdm
from time import time

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# dataset = CustomImageDataset('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/label.csv', '/home/sunwoo/Documents/Dataset/resize_image (copy)',transform=transform)
# load = torch.utils.data.DataLoader(dataset, shuffle=False)
# import os
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data.dataset import Dataset
# from tqdm.notebook import tqdm
# from time import time
#
# N_CHANNELS = 3
#
# dataset = CustomImageDataset('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/label.csv', '/home/sunwoo/Documents/Dataset/resize_image (copy)',transform=transform)
#
# full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())
#
# before = time()
# mean = torch.zeros(1)
# std = torch.zeros(1)
# print('==> Computing mean and std..')
# for inputs, _, _ in tqdm(full_loader):
#     temp_mean = torch.mean(inputs)
#     temp_std = torch.std(inputs)
#     mean += temp_mean
#     std += temp_std
# mean.div_(len(dataset))
# std.div_(len(dataset))
# print(mean, std)
#
# print("time elapsed: ", time()-before)
