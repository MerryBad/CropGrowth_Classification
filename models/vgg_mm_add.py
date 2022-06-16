import torch
import torch.nn as nn
import torchvision

cfg_base = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M'],  # , 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

cfg_major = [512, 512, 'M']

cfg_minor = [512, 512, 512, 'M']


class VGG_MM(nn.Module):

    def __init__(self, features, num_class=50, num_major=15):
        super().__init__()
        self.features = features[0]
        self.major_branch = features[1]
        self.minor_branch = features[2]
        self.major_avgpool = features[3]
        self.minor_avgpool = features[4]
        self.classifier_major = nn.Sequential(
            # nn.Conv2d(12044, num_major, kernel_size=1)
            nn.Linear(512*7*7, num_major),
        )

        self.classifier_minor = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x_major = self.major_branch(x)
        x_major = self.major_avgpool(x_major)
        x_major = torch.flatten(x_major, 1)
        pred_major = self.classifier_major(x_major)
        pred_major = pred_major.view(pred_major.size()[0], -1)

        # import pdb; pdb.set_trace()

        x = self.minor_branch(x)
        x = self.minor_avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(x.size()[0], -1)
        x_major = x_major.view(x_major.size()[0], -1)
        x += x_major
        pred_minor = self.classifier_minor(x)
        # import pdb; pdb.set_trace()
        # output = output.view(output.size()[0], -1)
        # output = self.classifier(output)
        # import pdb; pdb.set_trace()

        return pred_major, pred_minor


def make_layers(cfg, batch_norm=False):
    layers = nn.Sequential(*list(torchvision.models.vgg16_bn(pretrained=True).children())[0][:-10]).train()
    for p in layers.parameters():
        p.requires_grad = False

    major_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)))
    minor_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)))

    bch_major = nn.Sequential(
        nn.Conv2d(512, cfg_major[0], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_major[0]),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg_major[0], cfg_major[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_major[1]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.AdaptiveAvgPool2d(output_size=(5, 5))
    )

    bch_minor = nn.Sequential(
        nn.Conv2d(512, cfg_minor[0], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_minor[0]),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg_minor[0], cfg_minor[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_minor[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(cfg_minor[1], cfg_minor[2], kernel_size=3, padding=1),
        nn.BatchNorm2d(cfg_minor[2]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.AdaptiveAvgPool2d(output_size=(7,7))
    )

    return [layers, bch_major, bch_minor, major_avg_pool, minor_avg_pool]


def vgg16_bn(num_cls=50, num_mj=15):
    # print(VGG_MM(make_layers(cfg_base['D'], batch_norm=True), num_class=num_cls, num_major=num_mj))
    return VGG_MM(make_layers(cfg_base['D'], batch_norm=True), num_class=num_cls, num_major=num_mj)
