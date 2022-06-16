# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

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
from utils import WarmUpLR, get_network
from tqdm.notebook import tqdm
import time
from torchsummaryX import summary
from sklearn.model_selection import StratifiedShuffleSplit
# 경고무시를 위한 코드
import warnings

def train(epoch):

    net.train()
    for batch_index, (images, _, labels) in enumerate(training_loader):


        images = images.cuda()
        labels = torch.LongTensor(labels).squeeze(dim=-1).cuda()
        optimizer.zero_grad()
        outputs = net(images)
        #import pdb; pdb.set_trace()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        if epoch <= args.warm:
            warmup_scheduler.step()
        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index,
            total_samples=len(training_loader)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, _, labels) in test_loader:

        images = images.cuda()
        labels = torch.LongTensor(labels).squeeze(dim=-1).cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Valid set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader),
        correct.float() / len(test_loader)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Valid/Average loss', test_loss / len(test_loader), epoch)
    writer.add_scalar('Valid/Accuracy', correct.float() / len(test_loader), epoch)

    return correct.float() / len(test_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=False, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
    mean = 0.4148
    std = 0.2254
    #
    # # random_seed = 42
    # # transform = transforms.Compose([
    # #     transforms.ToPILImage(),
    # #     # transforms.Resize(32),
    # #     transforms.RandomResizedCrop(224),
    # #     transforms.RandomRotation(180),
    # #     transforms.RandomHorizontalFlip(),
    # #     transforms.RandomVerticalFlip(),
    # #     transforms.ToTensor(),
    # #     transforms.Normalize(mean=mean, std=std)
    # # ])
    transform = albumentations.Compose([
        albumentations.RandomResizedCrop(224, 224),
        # albumentations.OneOf([
        #     albumentations.RandomRotate90(p=0.5),
        #     albumentations.HorizontalFlip(p=0.5),
        #     albumentations.VerticalFlip(p=0.5),
        # ], p=1),
        albumentations.Normalize(mean=mean, std=std, p=1.0),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
    dataset = CustomImageDataset('label.csv', '/home/sunwoo/Documents/Dataset/resize_image (copy)/',
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
    print(len(train_indices))
    print(len(val_indices))
    print(len(test_indices))
    #
    # train_indices, val_indices = indices[split:], indices[:split]
    #
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    #
    # data preprocessing:
    training_loader = torch.utils.data.DataLoader(dataset,
                                                  num_workers=args.w,
                                                  batch_size=args.b,
                                                  shuffle=args.s,
                                                  sampler=train_sampler,
                                                  drop_last=True,
                                                  # pin_memory=True,
                                                  prefetch_factor=4
                                                  )
    test_loader = torch.utils.data.DataLoader(dataset,
                                              num_workers=args.w,
                                              batch_size=args.b,
                                              shuffle=args.s,
                                              sampler=val_sampler,
                                              drop_last=True,
                                              # pin_memory=True,
                                              prefetch_factor=4
                                              )
    #data preprocessing:

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(64, 3, 224, 224).cuda()
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
