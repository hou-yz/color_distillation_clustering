import numpy as np
import torch
import torch.nn as nn
from color_distillation.models.cyclegan import GeneratorResNet
from color_distillation.models.dncnn import DnCNN
from color_distillation.models.unet import UNet


class ColorCNN(nn.Module):
    def __init__(self, arch, temperature=1, color_norm=1, color_jitter=0):
        super().__init__()
        self.temperature = temperature
        self.color_norm = color_norm
        self.color_jitter = color_jitter
        if arch == 'unet':
            self.base = UNet()
        elif arch == 'dncnn':
            self.base = DnCNN()
        elif arch == 'cyclegan':
            self.base = GeneratorResNet()
        else:
            raise Exception
        self.color_mask = nn.Sequential(nn.Conv2d(self.base.out_channel, 256, 1), nn.ReLU(),
                                        nn.Conv2d(256, 256, 1, bias=False))
        self.mask_softmax = nn.Softmax2d()

    def forward(self, img, num_colors, training=True):
        feat = self.base(img)
        m = self.color_mask(feat)
        m = m[:, :num_colors]
        m = self.mask_softmax(self.temperature * m)  # softmax output
        M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map
        if training:
            color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8) / self.color_norm
            jitter_color_palette = color_palette + self.color_jitter * np.random.randn()
            transformed_img = (m.unsqueeze(1) * jitter_color_palette).sum(dim=2)
        else:
            indicator_M = torch.zeros_like(m).scatter(1, M, 1)
            color_palette = (img.unsqueeze(2) * indicator_M.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    indicator_M.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (indicator_M.unsqueeze(1) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette


def test():
    img = torch.randn([1, 3, 32, 32])
    model = ColorCNN('cyclegan', 128)
    train_img = model(img, 8)
    test_img = model(img, 8, training=False)
    pass


test()
