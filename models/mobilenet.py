from torch.nn.modules.activation import Hardswish
from torchvision.models import mobilenet_v3_small
import torch
import torch.nn as nn


class baseModel(nn.Module):
    def __init__(self):
        super(baseModel, self).__init__()

    def forward(x):
        pass


def simple_conv(inp, oup, kh, kw, padding=0, stride=1, groups=1, activation=nn.ReLU6, isBN=True, bias=False):
    modu = [nn.Conv2d(inp, oup, (kh, kw), stride, bias=bias,
                      groups=groups, padding=padding),
            ]
    if isBN:
        modu.append(nn.BatchNorm2d(oup))
    if activation is not None:
        modu.append(activation(inplace=True))
    return nn.Sequential(*modu)


class SELayer(baseModel):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Conv2d(in_channels, squeeze_channels, 1),
                                nn.BatchNorm2d(squeeze_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(squeeze_channels,
                                          in_channels, 1),
                                nn.BatchNorm2d(in_channels),
                                nn.Hardsigmoid(inplace=True))

    def forward(self, x):
        return x*self.se(self.avg_pool(x))


def make_divisible(v: float, divisor: int, min_value=None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class bottleNect(baseModel):
    def __init__(self, in_c, expand_size, out_c, kw, kh, act, stride, isSE):
        super().__init__()
        self.stride = stride
        assert act in ["RE", "HS", "None"]
        self.act = None
        if act == "RE":
            self.act = nn.ReLU
        elif act == "HS":
            self.act = nn.Hardswish
        # 1x1 conv
        self.conv1 = None
        if expand_size != in_c:
            self.conv1 = simple_conv(
                in_c, expand_size, 1, 1, activation=self.act)
        # dw convo
        self.conv2 = simple_conv(expand_size, expand_size, kh, kw, padding=(kh//2, kw//2),
                                 groups=expand_size, activation=self.act, stride=stride)
        self.conv3 = simple_conv(expand_size, out_c, 1, 1, padding=0,
                                 groups=1)
        self.SE = None
        if isSE:
            squeeze_channels = make_divisible(expand_size // 4, 8)
            self.SE = SELayer(expand_size, squeeze_channels)

        self.shortcut = in_c == out_c and stride == 1

    def forward(self, x):
        result = x
        if self.conv1 is not None:
            result = self.conv1(x)
        result = self.conv2(result)
        if self.SE is not None:
            result = self.SE(result)
        result = self.conv3(result)
        return result + x if self.shortcut else result


class mobilenetV3_Lite(baseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = simple_conv(
            3, 16, 3, 3, stride=2, activation=nn.Hardswish, padding=1)
        self.bnect1 = bottleNect(
            16, 16, 16, 3, 3, stride=2, isSE=True, act="RE")
        self.bnect2 = bottleNect(
            16, 72, 24, 3, 3, stride=2, isSE=False, act="RE")
        self.bnect3 = bottleNect(
            24, 88, 24, 3, 3, stride=1, isSE=False, act="RE")
        self.bnect4 = bottleNect(
            24, 96, 40, 5, 5, stride=2, isSE=True, act="HS")
        self.bnect5 = bottleNect(
            40, 240, 40, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect6 = bottleNect(
            40, 240, 40, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect7 = bottleNect(
            40, 120, 48, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect8 = bottleNect(
            48, 144, 48, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect9 = bottleNect(
            48, 288, 96, 5, 5, stride=2, isSE=True, act="HS")
        self.bnect10 = bottleNect(
            96, 576, 96, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect11 = bottleNect(
            96, 576, 96, 5, 5, stride=1, isSE=True, act="HS")
        self.conv2 = simple_conv(96, 576, 1, 1, activation=nn.Hardswish)
        self.pool = nn.AvgPool2d(7, 1)
        self.conv3 = simple_conv(576, 1024, 1, 1, isBN=False, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnect1(x)
        x = self.bnect2(x)
        x = self.bnect3(x)
        x = self.bnect4(x)
        x = self.bnect5(x)
        x = self.bnect6(x)
        x = self.bnect7(x)
        x = self.bnect8(x)
        x = self.bnect9(x)
        x = self.bnect10(x)
        x = self.bnect11(x)
        x = self.conv2(x)
        x = self.pool(x)
        return self.conv3(x)


if __name__ == "__main__":
    #a = mobilenet_v3_small(pretrained=True, progress=True)
    #print([i for i in a.state_dict().keys()])
    from torchsummary import summary
    a = mobilenetV3_Lite().cuda()
    b = mobilenet_v3_small().cuda()
    summary(a, (3, 224, 224))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    summary(b, (3, 224, 224))
