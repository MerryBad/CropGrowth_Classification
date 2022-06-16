""" train network using pytorch
"""
import csv

import argparse
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import SubsetRandomSampler
from albumentations.pytorch import ToTensorV2
import albumentations
from conf import settings
from dataset import *
from utils import WarmUpLR
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
    elif args.net == 'convnext':
        from models.convnext import convnext_base
        net = convnext_base()
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


def train(epoch):
    net.train()
    alpha = 0.3
    for batch_index, (images, super_class, finer_class) in enumerate(training_loader):
        images = images.cuda()
        optimizer.zero_grad()
        outputs_major, outputs_minor = net(images)

        labels_major = torch.LongTensor(super_class).squeeze(dim=-1).cuda()
        labels_minor = torch.LongTensor(finer_class).squeeze(dim=-1).cuda()

        # import pdb; pdb.set_trace()
        loss = alpha * loss_function(outputs_major, labels_major) + (1 - alpha) * loss_function(outputs_minor,
                                                                                                labels_minor)
        loss.backward()
        optimizer.step()
        if epoch <= args.warm:
            warmup_scheduler.step()
        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index,
            total_samples=len(training_loader)
        ))
        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    # print('epoch: {epoch}, lr: {:0.8f}'.format(optimizer.param_groups[0]['lr'], epoch=epoch))
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch):
    net.eval()
    alpha = 0.3
    test_loss = 0.0  # cost function error
    correct = 0.0
    correct_major = 0.0
    correct_minor = 0.0
    incorrect = 0
    mis_cls = 0.0
    mis_minor_win = 0.0
    mis_major_win = 0.0
    temp = 0
    for (images, super_class, finer_class) in validation_loader:
        # images = Variable(images)
        # labels = Variable(labels)

        images = images.cuda()
        # labels = labels.cuda()

        outputs_major, outputs_minor = net(images)

        labels_major = torch.LongTensor(super_class).squeeze(dim=-1).cuda()
        labels_minor = torch.LongTensor(finer_class).squeeze(dim=-1).cuda()

        loss = alpha * loss_function(outputs_major, labels_major) + (1 - alpha) * loss_function(outputs_minor,
                                                                                                labels_minor)

        test_loss += loss.item()

        _, preds_major = outputs_major.max(1)
        _, preds_minor = outputs_minor.max(1)
        preds_minor = preds_minor.cpu().numpy()
        preds_major = preds_major.cpu().numpy()
        # print(preds_major, ' ', preds_minor)
        # import pdb; pdb.set_trace()

        for i in range(len(preds_minor)):
            # 하위 클래스 레이블링이 '상위클래스 작물단계'로 구성, split 앞의 상위클래스만 추출 후 비교
            miss_match = (major_name[preds_major[i]] != (minor_name[preds_minor[i]].split()[0]))
            mis_cls += miss_match  # miss match 카운트
            if miss_match:  # 하위클래스의 예측값이 속하는 상위클래스와 에측한 상위클래스가 다른 경우에
                mis_minor_win += (preds_minor[i] == labels_minor[i])  # 하위 클래스를 맞춘 경우
                mis_major_win += (preds_major[i] == labels_major[i])  # 상위 클래스를 맞춘 경우

            correct_minor += (preds_minor[i] == labels_minor[i])  # 하위 클래스를 맞춘 경우
            correct_major += (preds_major[i] == labels_major[i])  # 상위 클래스를 맞춘 경우
            incorrect += (preds_major[i] == labels_major[i]) and (preds_minor[i] == labels_minor[i])
            if preds_minor[i] == labels_minor[i] and preds_major[i] == labels_major[i]:
                correct += 1.0  # 둘 다 맞춘 경우
            temp += 1
    print('Valid set Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        # test_loss / temp,
        test_loss / len(validation_loader),
        correct / temp
    ))
    print('Valid set major class acc: {:.4f}'.format(
        correct_major / temp
    ))
    print('Vaild set direct minor class acc: {:.4f}'.format(
        correct_minor / temp
    ))
    print('Vaild set missmatch acc: {:.4f}, major correct: {:.4f}, minor correct: {:.4f}, incorrect: {:.4f}'
          .format(mis_cls / temp, mis_major_win / mis_cls, mis_minor_win / mis_cls, incorrect / temp))

    # add informations to tensorboard
    writer.add_scalar('Valid/Missmatch', mis_cls / temp, epoch)
    writer.add_scalar('Valid/Average loss', test_loss / len(validation_loader), epoch)
    writer.add_scalar('Valid/Accuracy', correct / temp, epoch)
    writer.add_scalar('Valid/Major Accuracy', correct_major / temp, epoch)
    writer.add_scalar('Valid/Minor Accuracy', correct_minor / temp, epoch)

    return correct / temp, test_loss / temp


def test():
    net.eval()
    major_correct_1 = 0.0
    minor_correct_1 = 0.0
    major_correct_3 = 0.0
    minor_correct_3 = 0.0
    correct = 0.0
    temp = 0
    for n_iter, (images, super_class, finer_class) in enumerate(test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
        images = images.cuda()
        outputs_major, outputs_minor = net(images)

        labels_major = torch.LongTensor(super_class).squeeze(dim=-1).cuda()
        labels_minor = torch.LongTensor(finer_class).squeeze(dim=-1).cuda()

        _, top1_preds_major = outputs_major.max(1)
        _, top3_preds_major = outputs_major.topk(3, 1, largest=True, sorted=True)
        _, top1_preds_minor = outputs_minor.max(1)
        _, top3_preds_minor = outputs_minor.topk(3, 1, largest=True, sorted=True)
        top1_preds_minor = top1_preds_minor.cpu().numpy()
        # top3_preds_minor = top3_preds_minor.cpu().numpy()
        top1_preds_major = top1_preds_major.cpu().numpy()
        # top3_preds_major = top3_preds_major.cpu().numpy()
        # print(preds_major, ' ', preds_minor)
        # import pdb; pdb.set_trace()
        # print(top1_preds_minor)
        # print(top2_preds_minor)
        # print()
        # print(labels_major)
        # print(top1_preds_major)

        for i in range(len(top1_preds_minor)):
            # 작물 종류와 생육 단계에서 예측한 작물 종류가 같은지 확인
            # print(major_name[top1_preds_major[i]], minor_name[top1_preds_minor[i]].split()[0])
            # 작물 종류와 생육 단계 모두 맞춘 경우를 acc로 계산할 것
            correct += True if (top1_preds_major[i] == labels_major[i]) and (
                    top1_preds_minor[i] == labels_minor[i]) else False
            # 예측 값과 실제 값이 같은가?
            major_correct_1 += True if (top1_preds_major[i] == labels_major[i]) else False
            # print(True if (top1_preds_major[i] == labels_major[i]) else False)
            # print(major_correct_1)
            minor_correct_1 += True if (top1_preds_minor[i] == labels_minor[i]) else False
            # top 3 리스트 안에 정답이 있는가?
            major_correct_3 += True if labels_major[i] in top3_preds_major[i] else False
            minor_correct_3 += True if labels_minor[i] in top3_preds_minor[i] else False
            temp += 1
    return correct, major_correct_1, minor_correct_1, major_correct_3, minor_correct_3, temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=False, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    args = parser.parse_args()
    #
    net = get_network(args, use_gpu=args.gpu)
    net.load_state_dict(torch.load('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/checkpoint'
                                   '/convnext_base/2022-06-16T14:10:40.527677/convnext_base-40-regular.pth'))

    # net.eval()
    # print(net)

    # # print(summary(net, torch.zeros(64, 3, 224, 224).cuda()))
    mean = 0.4148
    std = 0.2254
    #
    # # random_seed = 42
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     # transforms.Resize(32),
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomRotation(180),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])
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
    # dataset_size = len(dataset)
    #
    # indices = list(range(dataset_size))
    # label_indices = []
    # for index in range(dataset_size):
    #     label_indices.append(int(dataset.__getitem__(index)[2]))
    #     # print(dataset.__getitem__(index)[2])
    #
    # sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
    # train_indices = []
    # val_indices = []
    # test_indices = []
    # for train_idx, test_idx in sss.split(np.zeros(len(label_indices)), label_indices):
    #     train_indices.append(train_idx)
    #     test_indices.append(test_idx)
    # train_indices = train_indices[0].tolist()
    # test_indices = test_indices[0].tolist()
    # sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=42)
    # label_indices = []
    # for index in range(len(test_indices)):
    #     label_indices.append(int(dataset.__getitem__(test_indices[index])[2]))
    # # print(label_indices[:10])
    # test_indices = []
    # for val_idx, test_idx in sss1.split(np.zeros(len(label_indices)), label_indices):
    #     val_indices.append(val_idx)
    #     test_indices.append(test_idx)
    #
    # val_indices = val_indices[0].tolist()
    # test_indices = test_indices[0].tolist()
    # print(len(train_indices))
    # print(len(val_indices))
    # print(len(test_indices))
    #
    # train_indices, val_indices = indices[split:], indices[:split]
    #

    data = list()
    f = open('/home/sunwoo/Documents/SGNet-master/pytorch-classification-SGNet/label_indices.csv', 'r')
    rea = csv.reader(f)
    for row in rea:
        data.append(row)
    f.close()
    data[0] = [int(i) for i in data[0]]
    data[1] = [int(i) for i in data[1]]
    data[2] = [int(i) for i in data[2]]

    train_sampler = SubsetRandomSampler(data[0])
    val_sampler = SubsetRandomSampler(data[1])
    test_sampler = SubsetRandomSampler(data[2])
    #
    # # data preprocessing:
    training_loader = torch.utils.data.DataLoader(dataset,
                                                  num_workers=args.w,
                                                  batch_size=args.b,
                                                  shuffle=args.s,
                                                  sampler=train_sampler,
                                                  drop_last=True,
                                                  # pin_memory=True,
                                                  prefetch_factor=4
                                                  )
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    num_workers=args.w,
                                                    batch_size=args.b,
                                                    shuffle=args.s,
                                                    sampler=val_sampler,
                                                    drop_last=True,
                                                    # pin_memory=True,
                                                    prefetch_factor=4
                                                    )
    test_loader = torch.utils.data.DataLoader(dataset,
                                              num_workers=args.w,
                                              batch_size=args.b,
                                              shuffle=args.s,
                                              sampler=test_sampler,
                                              drop_last=True,
                                              # pin_memory=True,
                                              prefetch_factor=4
                                              )
    # print(len(training_loader))
    # print(len(validation_loader))
    # print(len(test_loader))
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(32, 224, 4, 4).cuda()
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        train(epoch)
        acc, loss = eval_training(epoch)
        # start to save the best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    #

    correct, top1_major, top1_minor, top3_major, top3_minor, temp = test()
    # print(temp)
    # print(len(test_loader))
    # print(len(test_sampler))
    # print(len(test_indices))
    print("Coorect : ", correct / temp)
    print("Top major 1 acc: ", top1_major / temp)
    print("Top minor 1 acc: ", top1_minor / temp)
    print("Top major 3 acc: ", top3_major / temp)
    print("Top minor 3 acc: ", top3_minor / temp)
    # writer.close()
