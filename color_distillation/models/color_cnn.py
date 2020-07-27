import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            self.base_global = UNet(feature_dim=64)
        elif arch == 'dncnn':
            self.base_global = DnCNN()
        elif arch == 'cyclegan':
            self.base_global = GeneratorResNet()
        else:
            raise Exception
        bottleneck_channel = 3
        self.base_head = nn.Sequential(nn.Conv2d(self.base_global.out_channel, bottleneck_channel, 1),
                                       nn.BatchNorm2d(bottleneck_channel), nn.ReLU(), )
        self.base_attention = nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.ReLU(),
                                            nn.Linear(64, 2), nn.Softmax(dim=1))
        # support color quantization into 256 colors at most
        self.color_mask = nn.Sequential(nn.Conv2d(bottleneck_channel, 256, 1, bias=False))
        # self.mask_softmax = nn.Softmax2d()

    def forward(self, img, num_colors, mode='train'):
        B, _, H, W = img.shape
        feat = self.base_head(self.base_global(img))
        # feat = torch.cat([feat, img], dim=1)
        # global_local_weight = self.base_attention(torch.ones([B, 1]).to(img.device) * num_colors)
        # feat = self.base_head(self.base_global(img)) * global_local_weight[:, 0].view([B, 1, 1, 1]) + \
        #        img * global_local_weight[:, 1].view([B, 1, 1, 1])
        m = self.color_mask(feat)
        m = m.view([B, -1, num_colors, H, W]).sum(dim=1)
        m = F.softmax(self.temperature * m, dim=1)  # softmax output
        M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map
        if mode == 'train':
            color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8) / self.color_norm
            jitter_color_palette = color_palette + self.color_jitter * \
                                   torch.randn([B, 3, 1, 1, 1]).to(color_palette.device)
            transformed_img = (m.unsqueeze(1) * jitter_color_palette).sum(dim=2)
        else:
            if 'cluster' in mode:
                feat_1d = feat.permute([0, 2, 3, 1]).view([B, H * W, -1]).detach()
                for b in range(B):
                    # x = feat_1d[b]
                    # x_norm = (x ** 2).sum(1).view(-1, 1)
                    # dist = torch.clamp(x_norm + x_norm.permute([1, 0]) - 2.0 * torch.mm(x, x.permute([1, 0])), 0.0,
                    #                    np.inf)
                    labels = KMeans(n_clusters=num_colors, n_init=1).fit_predict(feat_1d[b].cpu().numpy())
                    M[b] = torch.from_numpy(labels.reshape([1, H, W]))
            indicator_M = torch.zeros_like(m).scatter(1, M, 1)
            color_palette = (img.unsqueeze(2) * indicator_M.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    indicator_M.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            transformed_img = (indicator_M.unsqueeze(1) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette


if __name__ == '__main__':
    img = torch.randn([10, 3, 32, 32])
    model = ColorCNN('cyclegan', 128)
    train_img = model(img, 8)
    test_img = model(img, 8, mode='test_cluster')
    pass
