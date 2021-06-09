import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_dim, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_dim)
        self.style = nn.Linear(style_dim, in_dim * 2)
        # init
        self.style.bias.data[:in_dim] = 1
        self.style.bias.data[in_dim:] = 0

    def forward(self, input, style):
        B, C, H, W = input.shape
        style = self.style(style).view([B, C * 2, 1, 1])
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


class ConvAdaINBlock(nn.Module):
    def __init__(self, in_dim, out_dim, style_dim):
        super(ConvAdaINBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_dim, out_dim, 3))
        self.adain1 = AdaptiveInstanceNorm(out_dim, style_dim)
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(out_dim, out_dim, 3))
        self.adain2 = AdaptiveInstanceNorm(out_dim, style_dim)

    def forward(self, x, style):
        out = F.relu(self.adain1(self.conv1(x), style))
        out = self.adain2(self.conv2(out), style)
        return out


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_dim, out_dim):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(),
                                  nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(inconv, self).__init__()
        self.conv = double_conv(in_dim, out_dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_dim, out_dim, style_dim):
        super(down, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.style_block = ConvAdaINBlock(in_dim, out_dim, style_dim)

    def forward(self, x, style):
        x = self.down(x)
        x = self.style_block(x, style)
        return x


class up(nn.Module):
    def __init__(self, in_dim, out_dim, style_dim):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.style_block = ConvAdaINBlock(in_dim, out_dim, style_dim)

    def forward(self, x1, x2, style):
        x1 = self.up(x1)
        diffX, diffY = x2.size()[3] - x1.size()[3], x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.style_block(x, style)
        return x


class outconv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class StyleUNet(nn.Module):
    def __init__(self, feature_dim=64, style_dim=256, max_color_num=8):
        super(StyleUNet, self).__init__()
        self.max_color_num = max_color_num
        self.color_style = nn.Sequential(nn.Linear(max_color_num, style_dim), nn.BatchNorm1d(style_dim), nn.ReLU(),
                                         nn.Linear(style_dim, style_dim), nn.BatchNorm1d(style_dim), nn.ReLU(),
                                         nn.Linear(style_dim, style_dim), nn.BatchNorm1d(style_dim), nn.ReLU(), )

        self.inc = inconv(3, feature_dim)
        self.down1 = down(feature_dim, feature_dim * 2, style_dim)
        self.down2 = down(feature_dim * 2, feature_dim * 4, style_dim)
        self.down3 = down(feature_dim * 4, feature_dim * 8, style_dim)
        self.down4 = down(feature_dim * 8, feature_dim * 8, style_dim)
        self.up1 = up(feature_dim * 16, feature_dim * 4, style_dim)
        self.up2 = up(feature_dim * 8, feature_dim * 2, style_dim)
        self.up3 = up(feature_dim * 4, feature_dim, style_dim)
        self.up4 = up(feature_dim * 2, feature_dim, style_dim)
        self.out_channel = feature_dim

    def forward(self, x, color):
        B, C, H, W = x.shape
        color_style = torch.zeros([B, self.max_color_num]).to(x.device).scatter(1, color.view([B, 1]), 1)
        color_style = self.color_style(color_style)
        x1 = self.inc(x)
        x2 = self.down1(x1, color_style)
        x3 = self.down2(x2, color_style)
        x4 = self.down3(x3, color_style)
        x5 = self.down4(x4, color_style)
        x = self.up1(x5, x4, color_style)
        x = self.up2(x, x3, color_style)
        x = self.up3(x, x2, color_style)
        x = self.up4(x, x1, color_style)
        return x


if __name__ == '__main__':
    B, C, H, W = 10, 3, 32, 32
    net = StyleUNet()
    color = torch.zeros([B]).long()
    x = torch.zeros([B, C, H, W])
    feat = net(x, color)
