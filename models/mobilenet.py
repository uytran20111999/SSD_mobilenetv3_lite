from torchvision.models import mobilenet_v3_small
import torch
import torch.nn as nn


class baseModel(nn.Module):
    def __init__(self):
        super(baseModel).__init__()

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
    def __init__(self, in_channels, r_factor=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Conv2d(in_channels, in_channels//r_factor, 1),
                                nn.BatchNorm2d(in_channels // r_factor),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels//r_factor,
                                          in_channels, 1),
                                nn.BatchNorm2d(in_channels),
                                nn.Hardsigmoid(inplace=True))

    def forward(self, x):
        return x*self.se(self.avg_pool(x))


class bottleNect(baseModel):
    def __init__(self, in_c, expand_size, out_c, kw, kh, act, stride, isSE):
        super().__init__()
        assert act in ["RE", "HW", "None"]
        self.act = None
        if act == "RE":
            self.act = nn.ReLU6
        elif act == "HW":
            self.act = nn.Hardswish
        # 1x1 conv
        self.conv1 = simple_conv(in_c, expand_size, 1, 1, activation=self.act)
        # dw convo
        self.conv2 = simple_conv(expand_size, expand_size, kh, kw, padding=(kh//2, kw//2),
                                 groups=expand_size, activation=self.act, stride=stride)
        self.conv3 = simple_conv(expand_size, out_c, 1, 1, padding=0,
                                 groups=1)
        self.SE = None
        if isSE:
            self.SE = SELayer(expand_size)

        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(simple_conv(in_c, out_c, 1, 1))

    def forward(self, x):
        result = self.conv1(x)
        sc = self.shortcut(x)
        result = self.conv2(result)
        result = self.conv3(result)
        if self.SE is not None:
            result = self.SE(result)
        return result + sc

        # 1 x 1 conv

    def forward(self, x):
        return self.moduls(x)


class mobilenetV3_Lite(baseModel):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    a = mobilenet_v3_small(pretrained=True, progress=True)
    #print([i for i in a.state_dict().keys()])
