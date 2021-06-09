import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from color_distillation.models.cyclegan import GeneratorResNet
from color_distillation.models.dncnn import DnCNN
from color_distillation.models.unet import UNet
from color_distillation.models.styleunet import StyleUNet


class ColorCNN(nn.Module):
    def __init__(self, arch, temperature=1.0, bottleneck_channel=16, colors_channel=256, topk=4, agg='mean'):
        super().__init__()
        self.topk = topk
        self.temperature = temperature
        self.arch = arch
        if arch == 'unet':
            self.base = UNet(feature_dim=64)
        elif arch == 'dncnn':
            self.base = DnCNN()
        elif arch == 'cyclegan':
            self.base = GeneratorResNet()
        elif arch == 'styleunet':
            self.base = StyleUNet()
        else:
            raise Exception
        self.bottleneck = nn.Sequential(nn.Conv2d(self.base.out_channel, bottleneck_channel, 1),
                                        nn.BatchNorm2d(bottleneck_channel), nn.ReLU(), )
        # support color quantization into 256 colors at most
        self.color_mask = nn.Conv2d(bottleneck_channel, colors_channel, 1, bias=False)
        self.agg = agg

    def forward(self, img, num_colors, mode='train'):
        B, _, H, W = img.shape
        if 'style' not in self.arch:
            feat = self.bottleneck(self.base(img))
        else:
            feat = self.bottleneck(self.base(img, int(np.log2(num_colors) - 1) * torch.ones([B]).long().to(img.device)))
        m = self.color_mask(feat)
        if self.agg == 'mean':
            m = m.view([B, -1, num_colors, H, W]).mean(dim=1)
        elif self.agg == 'max':
            m, _ = m.view([B, -1, num_colors, H, W]).max(dim=1)
        else:
            raise Exception
        if mode == 'train':
            topk, idx = torch.topk(m, min(self.topk, num_colors), dim=1)
            m = torch.scatter(torch.zeros_like(m), 1, idx, F.softmax(topk / self.temperature, dim=1))  # softmax output
            color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (m.unsqueeze(1) * color_palette).sum(dim=2)
        else:
            m = F.softmax(m / self.temperature, dim=1)
            M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map
            M = torch.zeros_like(m).scatter(1, M, 1)
            color_palette = (img.unsqueeze(2) * M.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    M.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (M.unsqueeze(1) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette


if __name__ == '__main__':
    from color_distillation import datasets
    import color_distillation.utils.transforms as T
    from color_distillation.loss.similarity_preserving import *

    num_colors = 2
    dataset = datasets.STL10('/home/houyz/Data/stl10', split='test', color_quantize=T.MedianCut(),
                             transform=T.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=0, pin_memory=True)
    dataloader.dataset.num_colors[0] = num_colors
    img, target = next(iter(dataloader))
    img, label, quantized_img, index_map = img.cuda(), target[0].cuda(), target[1][0].cuda(), target[1][1].cuda()
    B, C, H, W = img.shape
    model = ColorCNN('styleunet').cuda()
    trans_img, m, color_palette = model(img, num_colors, mode='test')
    crop = T.RandomCrop(32, padding=4)
    crop(trans_img)
    pass
