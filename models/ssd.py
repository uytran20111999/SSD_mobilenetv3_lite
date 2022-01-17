from locale import normalize
from tokenize import group
from models.mobilenet import mobilenetV3_Lite, baseModel, simple_conv
from torchvision.models import mobilenet_v3_small  # for load pretrain
import torch.nn as nn
import torch


def extra_block(in_channels, out_channels, is_BN):
    activation = nn.ReLU6
    inter_channels = out_channels//2
    extras = nn.Sequential(simple_conv(in_channels, inter_channels, 1, 1, activation=activation, isBN=is_BN),
                           simple_conv(inter_channels, inter_channels, 3, 3,
                                       activation=activation, groups=inter_channels, stride=2, padding=1, isBN=is_BN),
                           simple_conv(inter_channels, out_channels, 1, 1, activation=activation, isBN=is_BN),)
    return extras


def normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


def prediction_block(in_channels, out_channels, kernel_size, is_BN):
    activation = nn.ReLU6
    padding = (kernel_size - 1) // 2
    return nn.Sequential(simple_conv(in_channels, in_channels, kernel_size,
                         kernel_size, groups=in_channels, isBN=is_BN, padding=padding, activation=activation),
                         simple_conv(in_channels, out_channels, 1, 1, isBN=False, activation=None))


class SSDLiteClassificationHead(baseModel):
    def __init__(self, in_channels, num_anchors, num_classes, is_BN):
        super().__init__()
        self.class_heads = nn.ModuleList()
        self.num_classes = num_classes
        for i, j in zip(in_channels, num_anchors):
            self.class_heads.append(prediction_block(
                in_channels=i, out_channels=j*num_classes, kernel_size=3, is_BN=is_BN))

    def forward(self, xs):
        # x [B x C1 x H1 x W1, BxC2xH2xW1]
        assert len(xs) == len(self.class_heads)
        ret = []
        for x, block in zip(xs, self.class_heads):
            temp = block(x)
            # B x num_classes x H x W
            ret.append(
                temp.view(temp.shape[0], self.num_classes, -1).contiguous().permute(0, 2, 1))
        return torch.cat(ret, dim=1)  # B x num_anchors x num_classes


class SSDLiteRegressorHead(baseModel):
    def __init__(self, in_channels, num_anchors, is_BN):
        super().__init__()
        self.reg_heads = nn.ModuleList()
        for i, j in zip(in_channels, num_anchors):
            self.reg_heads.append(prediction_block(
                in_channels=i, out_channels=j*4, kernel_size=3, is_BN=is_BN))

    def forward(self, xs):
        # x [B x C1 x H1 x W1, BxC2xH2xW1]
        assert len(xs) == len(self.reg_heads)
        ret = []
        for x, block in zip(xs, self.reg_heads):
            temp = block(x)
            ret.append(temp.view(
                temp.shape[0], 4, -1).contiguous().permute(0, 2, 1))  # B x 4 x (H x W x num_anchors_per_cell)
        return torch.cat(ret, dim=1)  # B x num_anchors x 4


class SSDLiteHead(baseModel):
    def __init__(self, in_channels, num_anchors, num_class, is_BN):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_class, is_BN)
        self.regressor_head = SSDLiteRegressorHead(
            in_channels, num_anchors, is_BN)
        normal_init(self)

    def forward(self, x):
        # x: list of tensors of shape B x C x H x W
        return {"classification": self.classification_head(x), "regression": self.regressor_head(x)}


class Mobilenetv3SmallExtra(baseModel):
    # to do: adds extras layers to mobilenet
    def __init__(self, base_models, c4_idx=9, is_BN=True):
        super().__init__()
        assert base_models.layers[c4_idx].conv1 is not None
        self.base_net = nn.ModuleList()
        self.base_net.append(nn.Sequential(
            *base_models.layers[:c4_idx], base_models.layers[c4_idx].conv1))  # c4
        self.base_net.append(nn.Sequential(
            *(list(base_models.layers[c4_idx].children())[1:]), *base_models.layers[c4_idx+1:]))  # c5
        self.extras = [
            extra_block(base_models.last_out_channels, 512, is_BN),
            extra_block(512, 256, is_BN),
            extra_block(256, 256, is_BN),
            extra_block(256, 128, is_BN)
        ]
        self.extras = nn.ModuleList(self.extras)
        normal_init(self.extras)

    def forward(self, x):
        # x: BxCxHxW
        results_blocks = []
        for block in self.base_net:
            x = block(x)
            results_blocks.append(x)
        for block in self.extras:
            x = block(x)
            results_blocks.append(x)
        return results_blocks

    def freeze_base(self):
        for module in self.base_net:
            for p in module.parameters():
                p.requires_grad_(False)

    def unfreeze_base(self):
        for module in self.base_net:
            for p in module.parameters():
                p.requires_grad_(True)


class SSDLite(baseModel):
    def __init__(self, is_pretrain=True, c4_idx=9, is_BN=True,
                 in_channels=[288, 576, 512, 256, 256, 128], num_bbxs=[3, 6, 6, 6, 6, 6], num_class=3):
        base_net_state_dict = None
        super().__init__()
        if is_pretrain:  # load the trained model from torch implementations
            base_net_state_dict = mobilenet_v3_small().features
            base_net_state_dict = base_net_state_dict.state_dict()

        self.mobilenet_small = mobilenetV3_Lite(
            pretrain_state_dict=base_net_state_dict)
        self.feature_extractor = Mobilenetv3SmallExtra(
            self.mobilenet_small, c4_idx, is_BN)
        if is_pretrain:
            self.feature_extractor.freeze_base()
        self.head = SSDLiteHead(in_channels, num_bbxs, num_class, is_BN)

    def forward(self, x):
        features_list = self.feature_extractor(x)
        return self.head(features_list)


if __name__ == "__main__":
    a = SSDLite().cuda()
    x = torch.randn((2, 3, 300, 300)).cuda()
    result = a(x)
    k = 5
