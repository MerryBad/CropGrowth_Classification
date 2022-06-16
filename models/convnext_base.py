import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class ConvNeXt(nn.Module):
    def __init__(self, features, num_class=50, num_major=15):
        super().__init__()
        self.features = features[0]
        self.major_branch = features[1]
        self.minor_branch = features[2]
        self.norm = nn.LayerNorm([1024], eps=1e-6)  # final norm layer
        self.avg_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.classifier_major = nn.Sequential(
            *list(torchvision.models.convnext_base(pretrained=False).children())[-1][:-1]).eval()
        self.classifier_major.add_module("Liear", nn.Linear(in_features=1024, out_features=num_major, bias=True))

        self.classifier_minor = nn.Sequential(nn.LayerNorm(2048, eps=1e-06, elementwise_affine=True),
                                              nn.Flatten(),
                                              nn.Linear(in_features=2048, out_features=num_class, bias=True))

    def forward(self, x):
        x = self.features(x)
        x_major = self.major_branch(x)
        x_major = self.avg_layer(x_major)
        pred_major = self.classifier_major(x_major)
        # pred_major = pred_major.view(pred_major.size()[0], -1)

        # import pdb; pdb.set_trace()

        x = self.minor_branch(x)
        x = self.avg_layer(x)
        # x = LayerNorm(x.mean([-2,-1]))
        # x = torch.flatten(x, 1)
        x = x.view(x.size()[0], -1)
        x_major = x_major.view(x_major.size()[0], -1)
        x = torch.cat((x, x_major), 1)
        pred_minor = self.classifier_minor(x)
        # import pdb; pdb.set_trace()
        # output = output.view(output.size()[0], -1)
        # output = self.classifier(output)
        # import pdb; pdb.set_trace()

        return pred_major, pred_minor


def make_layers():
    layers = nn.Sequential(*list(torchvision.models.convnext_base(pretrained=True).children())[:-2][0][:-1]).eval()
    for p in layers.parameters():
        p.requires_grad = False
    bch_major = nn.Sequential(
        *list(torchvision.models.convnext_base(pretrained=True).children())[:-2][0][-1][:-1]).eval()

    bch_minor = nn.Sequential(
        *list(torchvision.models.convnext_base(pretrained=True).children())[:-2][0][-1]).eval()

    return [layers, bch_major, bch_minor]


def ConvNext_base(num_cls=50, num_mj=15):
    return ConvNeXt(make_layers(), num_class=num_cls, num_major=num_mj)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
