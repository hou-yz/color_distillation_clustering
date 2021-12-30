import torch
import torch.nn as nn
import torch.nn.functional as F
from color_distillation.models.resnet_seg import resnet18


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, depth_sep=False):
        super(_ASPPModule, self).__init__()
        if not depth_sep or kernel_size == 1:
            conv = nn.Conv2d
        else:
            conv = DepthwiseSeparableConv
        self.atrous_conv = nn.Sequential(conv(inplanes, planes, kernel_size, 1, padding, dilation, bias=False),
                                         nn.BatchNorm2d(planes), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.atrous_conv(x)
        return x


class ASPP(nn.Module):
    def __init__(self, inplanes=2048, dilations=(1, 12, 24, 36), depth_sep=False, feat_dim=256):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(inplanes, feat_dim, 1, padding=0, dilation=dilations[0], depth_sep=depth_sep)
        self.aspp2 = _ASPPModule(inplanes, feat_dim, 3, padding=dilations[1], dilation=dilations[1],
                                 depth_sep=depth_sep)
        self.aspp3 = _ASPPModule(inplanes, feat_dim, 3, padding=dilations[2], dilation=dilations[2],
                                 depth_sep=depth_sep)
        self.aspp4 = _ASPPModule(inplanes, feat_dim, 3, padding=dilations[3], dilation=dilations[3],
                                 depth_sep=depth_sep)

        self.assp_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(inplanes, feat_dim, 1, bias=False),
                                       nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))
        self.joint_conv = nn.Sequential(nn.Conv2d(feat_dim * 5, feat_dim, 1, bias=False),
                                        nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True), )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.assp_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.joint_conv(x)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21, base='resnet18', output_stride=16, dropout=0.1,
                 depth_sep=True, **kwargs):
        super(DeepLabV3, self).__init__()
        if base == 'resnet18':
            self.base = resnet18(pretrained=True, output_stride=output_stride, **kwargs)
        else:
            raise Exception
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.low_level_conv = nn.Sequential(nn.Conv2d(64, 48, 1, bias=False),
                                            nn.BatchNorm2d(48), nn.ReLU(inplace=True), )
        self.aspp = ASPP(512, dilations, depth_sep)
        conv = nn.Conv2d if not depth_sep else DepthwiseSeparableConv
        self.joint_conv = nn.Sequential(conv(304, 256, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        conv(256, 256, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(256), nn.ReLU(inplace=True), )

        self.classifier = nn.Sequential(nn.Dropout2d(p=dropout), nn.Conv2d(256, num_classes, 1))

    def forward(self, x):
        feats = self.base(x)

        low_level_feat = self.low_level_conv(feats[0])
        x = self.aspp(feats[3])
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear')
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.joint_conv(x)

        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = DeepLabV3()
    img = torch.zeros([8, 3, 320, 320])
    ret = model(img)
    print(ret.shape)
    pass
