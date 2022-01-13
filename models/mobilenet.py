from torch.nn.modules.activation import Hardswish
from torchvision.models import mobilenet_v3_small
import torch
import torch.nn as nn


class baseModel(nn.Module):
    def __init__(self):
        super(baseModel, self).__init__()

    def forward(x):
        pass


def simple_conv(inp, oup, kh, kw, padding=0, stride=1, groups=1, activation=nn.ReLU, isBN=True, bias=False):
    modu = [nn.Conv2d(inp, oup, (kh, kw), stride, bias=bias,
                      groups=groups, padding=padding),
            ]
    if isBN:
        modu.append(nn.BatchNorm2d(oup, eps=0.001, momentum=0.01))
    if activation is not None:
        modu.append(activation(inplace=True))
    return nn.Sequential(*modu)


class SELayer(baseModel):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Conv2d(in_channels, squeeze_channels, 1),
                                # nn.BatchNorm2d(squeeze_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(squeeze_channels,
                                          in_channels, 1),
                                # nn.BatchNorm2d(in_channels),
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
        self.SE = None
        if isSE:
            squeeze_channels = make_divisible(expand_size // 4, 8)
            self.SE = SELayer(expand_size, squeeze_channels)
        self.conv3 = simple_conv(expand_size, out_c, 1, 1, padding=0,
                                 groups=1, activation=None)

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


def convert_from_torch_implement(state_dict_custom, torch_state_dict):
    # the code work for this specific case since the blocks' order is the same as torch's implementation.
    # work either for torch's mobilenetv3 small state dict or self state dict
    l1 = list(state_dict_custom.keys())
    l2 = list(torch_state_dict.keys())
    for i, j in zip(l1, l2):
        state_dict_custom[i] = torch_state_dict[j]
    return state_dict_custom


class mobilenetV3_Lite(baseModel):
    def __init__(self, pretrain_state_dict=None):
        super().__init__()
        self.pretrain_state_dict = pretrain_state_dict
        self.layers = nn.ModuleList()
        self.conv0 = simple_conv(
            3, 16, 3, 3, stride=2, activation=nn.Hardswish, padding=1)
        self.bnect0 = bottleNect(
            16, 16, 16, 3, 3, stride=2, isSE=True, act="RE")
        self.bnect1 = bottleNect(
            16, 72, 24, 3, 3, stride=2, isSE=False, act="RE")
        self.bnect2 = bottleNect(
            24, 88, 24, 3, 3, stride=1, isSE=False, act="RE")
        self.bnect3 = bottleNect(
            24, 96, 40, 5, 5, stride=2, isSE=True, act="HS")
        self.bnect4 = bottleNect(
            40, 240, 40, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect5 = bottleNect(
            40, 240, 40, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect6 = bottleNect(
            40, 120, 48, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect7 = bottleNect(
            48, 144, 48, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect8 = bottleNect(
            48, 288, 96, 5, 5, stride=2, isSE=True, act="HS")
        self.bnect9 = bottleNect(
            96, 576, 96, 5, 5, stride=1, isSE=True, act="HS")
        self.bnect10 = bottleNect(
            96, 576, 96, 5, 5, stride=1, isSE=True, act="HS")
        self.conv1 = simple_conv(96, 576, 1, 1, activation=nn.Hardswish)
        self.last_out_channels = 576
        self.layers.extend([self.conv0, self.bnect0, self.bnect1, self.bnect2, self.bnect3,
                            self.bnect4, self.bnect5, self.bnect6, self.bnect7, self.bnect8,
                            self.bnect9, self.bnect10, self.conv1])
        self.init_weight()

    def init_weight(self):
        if self.pretrain_state_dict is not None:
            load_std = convert_from_torch_implement(
                self.state_dict(), self.pretrain_state_dict)
            self.load_state_dict(load_std)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bnect0(x)  # C1
        x = self.bnect1(x)  # C2
        x = self.bnect2(x)
        x = self.bnect3(x)  # C3
        x = self.bnect4(x)
        x = self.bnect5(x)
        x = self.bnect6(x)
        x = self.bnect7(x)
        x = self.bnect8(x)  # C4
        x = self.bnect9(x)
        x = self.bnect10(x)
        x = self.conv1(x)  # C5
        return x

    def freeze_model(self):
        for module in self:
            for p in module.parameters():
                p.requires_grad_(False)

    def unfreeze_model(self):
        for module in self:
            for p in module.parameters():
                p.requires_grad_(True)


if __name__ == "__main__":
    from torchsummary import summary
    b = mobilenet_v3_small().features
    torch_model = b.state_dict()
    a = mobilenetV3_Lite(
        pretrain_state_dict=b.state_dict())
    k = 5
    a = a.eval().cuda()
    b = b.eval().cuda()
    x = torch.rand((2, 3, 224, 224)).cuda()
    y1 = a(x)
    y2 = b(x)
    print((y1-y2).sum())
