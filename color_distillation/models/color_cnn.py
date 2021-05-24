import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from color_distillation.models.cyclegan import GeneratorResNet
from color_distillation.models.dncnn import DnCNN
from color_distillation.models.unet import UNet


class ColorCNN(nn.Module):
    def __init__(self, arch, temperature=1.0, bottleneck_channel=16, topk=4):
        super().__init__()
        self.topk = topk
        self.temperature = temperature
        if arch == 'unet':
            self.base_global = UNet(feature_dim=64)
        elif arch == 'dncnn':
            self.base_global = DnCNN()
        elif arch == 'cyclegan':
            self.base_global = GeneratorResNet()
        else:
            raise Exception
        self.bottleneck = nn.Sequential(nn.Conv2d(self.base_global.out_channel, bottleneck_channel, 1),
                                        nn.BatchNorm2d(bottleneck_channel), nn.ReLU(), )
        # support color quantization into 256 colors at most
        self.color_mask = nn.Conv2d(bottleneck_channel, 256, 1, bias=False)

    def forward(self, img, num_colors, mode='train'):
        B, _, H, W = img.shape
        feat = self.bottleneck(self.base_global(img))
        m = self.color_mask(feat)
        m = m.view([B, -1, num_colors, H, W]).sum(dim=1)
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
    model = ColorCNN('unet').cuda()
    model.load_state_dict(torch.load(
        'logs/colorcnn/stl10/resnet18/-64colors/ce0.0_kd0.0_recons0.0_max1.0_pixsim10.0_conf1.0_info1.0_jit1.0_norm4.0_kd0.0_prcp0.0_neck16_topk4_2021-05-22_16-58-17/ColorCNN.pth'))
    trans_img, m, color_palette = model(img, num_colors, mode='test')
    pass
