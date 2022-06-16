""" train network using pytorch
"""
import csv

from torchsampler import ImbalancedDatasetSampler
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
from torchvision.transforms import transforms
from albumentations.pytorch import ToTensorV2
import albumentations
from conf import settings
from dataset import *
from utils import WarmUpLR
from tqdm.notebook import tqdm
import time
from torchsummaryX import summary
from sklearn.model_selection import StratifiedShuffleSplit
# 경고무시를 위한 코드
import warnings

warnings.filterwarnings(action='ignore')

NUM_CLS_MAJOR = 15
NUM_CLS_MINOR = 50
major_name = ['가지', '거베라', '고추(청양)', '국화', '딸기', '멜론', '부추', '상추', '안개', '애플망고', '애호박', '오이', '장미',
              '토마토(일반)', '파프리카(볼로키)']
minor_name = ['가지 개화기', '가지 수확기', '가지 영양생장', '거베라 수확기', '거베라 영양생장',
              '거베라 정식기', '고추(청양) 개화기', '고추(청양) 생육기', '고추(청양) 수확기', '고추(청양) 정식기',
              '국화 생육기', '국화 영양생장', '국화 정식기', '국화 화아발달기', '국화 화아분화기',
              '딸기 개화기', '딸기 꽃눈분화기', '딸기 수확기', '멜론 개화기', '멜론 과실비대기',
              '멜론 영양생장', '멜론 정식기', '부추 생육기', '부추 영양생장', '상추 생육기',
              '상추 수확기', '상추 영양생장', '상추 착과기', '안개 개화기', '안개 수확기',
              '안개 영양생장', '애플망고 개화기', '애플망고 화아분화기', '애호박 생육기', '애호박 수확기',
              '애호박 영양생장', '애호박 착과기', '오이 개화기', '오이 영양생장', '장미 영양생장',
              '장미 절화기', '토마토(일반) 생육기', '토마토(일반) 수확기', '토마토(일반) 정식기', '토마토(일반) 착과기',
              '파프리카(볼로키) 과비대성숙기', '파프리카(볼로키) 과실비대기', '파프리카(볼로키) 수확기', '파프리카(볼로키) 착과기', '파프리카(볼로키) 착색기']


def get_network(args, use_gpu=True):
    """ return given network
    """
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg_mm16':
        from models.vgg_mm import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg_mm16_add':
        from models.vgg_mm_add import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'convnext_base':
        from models.convnext_base import ConvNext_base
        net = ConvNext_base()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net

def test():
    net.eval()
    major_correct_1 = 0.0
    minor_correct_1 = 0.0
    major_correct_3 = 0.0
    minor_correct_3 = 0.0
    miss_match= 0.0
    mis_minor_win =0.0
    mis_major_win=0.0
    correct = 0.0
    incorrect=0.0
    temp = 0
    for n_iter, (images, super_class, finer_class) in enumerate(test_loader):
        # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
        images = images.cuda()
        outputs_major, outputs_minor = net(images)

        labels_major = torch.LongTensor(super_class).squeeze(dim=-1).cuda()
        labels_minor = torch.LongTensor(finer_class).squeeze(dim=-1).cuda()

        _, top1_preds_major = outputs_major.max(1)
        _, top3_preds_major = outputs_major.topk(3, 1, largest=True, sorted=True)
        _, top1_preds_minor = outputs_minor.max(1)
        _, top3_preds_minor = outputs_minor.topk(3, 1, largest=True, sorted=True)
        top1_preds_minor = top1_preds_minor.cpu().numpy()
        top1_preds_major = top1_preds_major.cpu().numpy()

        for i in range(len(top1_preds_minor)):
            miss = (major_name[top1_preds_major[i]] != (minor_name[top1_preds_minor[i]].split()[0]))
            miss_match += miss
            if miss:  # 하위클래스의 예측값이 속하는 상위클래스와 에측한 상위클래스가 다른 경우에
                mis_minor_win += (top1_preds_minor[i] == labels_minor[i])  # 하위 클래스를 맞춘 경우
                mis_major_win += (top1_preds_major[i] == labels_major[i])  # 상위 클래스를 맞춘 경우
            incorrect += (top1_preds_major[i] == labels_major[i]) and (top1_preds_minor[i] == labels_minor[i])
            # 작물 종류와 생육 단계 모두 맞춘 경우를 acc로 계산할 것
            correct += True if (top1_preds_major[i] == labels_major[i]) and (
                    top1_preds_minor[i] == labels_minor[i]) else False
            # 예측 값과 실제 값이 같은가?
            major_correct_1 += True if (top1_preds_major[i] == labels_major[i]) else False
            minor_correct_1 += True if (top1_preds_minor[i] == labels_minor[i]) else False
            # top 3 리스트 안에 정답이 있는가?
            major_correct_3 += True if labels_major[i] in top3_preds_major[i] else False
            minor_correct_3 += True if labels_minor[i] in top3_preds_minor[i] else False
            temp += 1
    print('missmatch acc: {:.4f}, major correct: {:.4f}, minor correct: {:.4f}, incorrect: {:.4f}'
          .format(miss_match / temp, mis_major_win / miss_match, mis_minor_win / miss_match, incorrect / temp))
    return correct, major_correct_1, minor_correct_1, major_correct_3, minor_correct_3, temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=False, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    args = parser.parse_args()
    #
    net = get_network(args, use_gpu=args.gpu)
    net.load_state_dict(torch.load('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/checkpoint/vgg_mm16_add/2022-06-13T18:12:50.995470/vgg_mm16_add-296-best.pth'))
    # # print(summary(net, torch.zeros(64, 3, 224, 224).cuda()))
    mean = 0.4148
    std = 0.2254
    transform = albumentations.Compose([
        albumentations.RandomResizedCrop(224, 224),
        albumentations.OneOf([
            albumentations.RandomRotate90(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
        ], p=1.0),
        albumentations.Normalize(mean=mean, std=std, p=1.0),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
    dataset = CustomImageDataset('label.csv', '/home/sunwoo/Documents/Dataset/resize_image_copy/',
                                 transform=transform)
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    label_indices = []
    for index in range(dataset_size):
        label_indices.append(int(dataset.__getitem__(index)[2]))
        # print(dataset.__getitem__(index)[2])

    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
    train_indices = []
    val_indices = []
    test_indices = []
    for train_idx, test_idx in sss.split(np.zeros(len(label_indices)), label_indices):
        train_indices.append(train_idx)
        test_indices.append(test_idx)
    train_indices = train_indices[0].tolist()
    test_indices = test_indices[0].tolist()
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=42)
    label_indices = []
    for index in range(len(test_indices)):
        label_indices.append(int(dataset.__getitem__(test_indices[index])[2]))
    # print(label_indices[:10])
    test_indices = []
    for val_idx, test_idx in sss1.split(np.zeros(len(label_indices)), label_indices):
        val_indices.append(val_idx)
        test_indices.append(test_idx)

    val_indices = val_indices[0].tolist()
    test_indices = test_indices[0].tolist()
    # indices = []
    # with open("label_indices.csv", 'r') as file:
    #     f = csv.reader(file)
    #     for line in f:
    #         indices.append(line)
    #
    #     train_indices = indices[0]
    #     val_indices=indices[1]
    #     test_indices=indices[2]
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    # with open("label_indices.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(train_indices)
    #     writer.writerow(val_indices)
    #     writer.writerow(test_indices)

    # data preprocessing:
    test_loader = torch.utils.data.DataLoader(dataset,
                                              num_workers=args.w,
                                              batch_size=args.b,
                                              shuffle=args.s,
                                              sampler=test_sampler,
                                              drop_last=True,
                                              # pin_memory=True,
                                              prefetch_factor=4
                                              )
    correct, top1_major, top1_minor, top3_major, top3_minor, temp = test()
    print("Coorect : ", correct / temp)
    print("Top major 1 acc: ", top1_major / temp)
    print("Top minor 1 acc: ", top1_minor / temp)
    print("Top major 3 acc: ", top3_major / temp)
    print("Top minor 3 acc: ", top3_minor / temp)
