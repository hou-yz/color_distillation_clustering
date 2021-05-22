import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BatchSimLoss(nn.Module):
    def __init__(self):
        super(BatchSimLoss, self).__init__()

    def forward(self, featmap_src, featmap_tgt):
        f_src, f_tgt = featmap_src.view([B, -1]), featmap_tgt.view([B, -1])
        A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss_sim = torch.norm(A_src - A_tgt) ** 2 / B
        return loss_sim


class PixelSimLoss(nn.Module):
    def __init__(self):
        super(PixelSimLoss, self).__init__()

    def forward(self, featmap_src, featmap_tgt):
        B, C, H, W = featmap_src.shape
        f_src, f_tgt = featmap_src.view([B, C, H * W]), featmap_tgt.view([B, C, H * W])
        A_src, A_tgt = torch.bmm(f_src.permute([0, 2, 1]), f_src), torch.bmm(f_tgt.permute([0, 2, 1]), f_tgt)
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss_semantic = torch.mean(torch.norm(A_src - A_tgt, dim=(1, 2)) ** 2 / (H * W))
        return loss_semantic


class PixelSampleSimLoss(nn.Module):
    def __init__(self, sample_ratio=0.3):
        super(PixelSampleSimLoss, self).__init__()
        self.sample_ratio = sample_ratio

    def forward(self, featmap_src, featmap_tgt):
        B, C, H, W = featmap_src.shape
        sample_idx = [np.random.choice(H * W, int(H * W * self.sample_ratio), replace=False) for _ in range(B)]
        sample_idx = np.stack(sample_idx, axis=0).reshape([B, 1, int(H * W * self.sample_ratio)]).repeat(C, axis=1)
        f_src, f_tgt = featmap_src.view([B, C, H * W]).gather(2, torch.from_numpy(sample_idx).to(featmap_src.device)), \
                       featmap_tgt.view([B, C, H * W]).gather(2, torch.from_numpy(sample_idx).to(featmap_tgt.device))
        A_src, A_tgt = torch.bmm(f_src.permute([0, 2, 1]), f_src), torch.bmm(f_tgt.permute([0, 2, 1]), f_tgt)
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss_semantic = torch.mean(torch.norm(A_src - A_tgt, dim=(1, 2)) ** 2 / (H * W))
        return loss_semantic


class ChannelSimLoss(nn.Module):
    def __init__(self):
        super(ChannelSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C = featmap_src_T.shape[:2]
        f_src, f_tgt = featmap_src_T.view([B, C, -1]), featmap_tgt_S.view([B, C, -1])
        A_src, A_tgt = torch.bmm(f_src, f_src.permute([0, 2, 1])), torch.bmm(f_tgt, f_tgt.permute([0, 2, 1]))
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss = torch.mean(torch.norm(A_src - A_tgt, dim=(1, 2)) ** 2 / C)
        return loss


if __name__ == '__main__':
    B, C, H, W = 32, 128, 10, 10
    feat1, feat2 = torch.randn([B, C, H, W]), torch.randn([B, C, H, W])
    batch_loss = BatchSimLoss()
    l1 = batch_loss(feat1, feat2)
    pixel_loss = PixelSimLoss()
    l2 = pixel_loss(feat1, feat2)
    pixel_loss = PixelSampleSimLoss()
    l2_2 = pixel_loss(feat1, feat2)
    channel_loss = ChannelSimLoss()
    l3 = channel_loss(feat1, feat2)

    feat1, feat2 = torch.randn([B, C]), torch.randn([B, C])
    l3_1 = channel_loss(feat1, feat2)
    pass
