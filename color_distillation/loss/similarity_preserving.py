import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BatchSimLoss(nn.Module):
    def __init__(self, normalize=True):
        super(BatchSimLoss, self).__init__()
        self.normalize = normalize

    def forward(self, featmap_src, featmap_tgt):
        B = featmap_src.shape[0]
        f_src, f_tgt = featmap_src.view([B, -1]), featmap_tgt.view([B, -1])
        A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
        if self.normalize:
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss_sim = torch.norm(A_src - A_tgt) ** 2 / B
        else:
            # loss_semantic = torch.mean(torch.norm(A_src - A_tgt, dim=(1, 2)) / sample_idx.shape[-1])
            loss_sim = F.binary_cross_entropy(A_src, A_tgt)
        return loss_sim


class PixelSimLoss(nn.Module):
    def __init__(self, sample_ratio=0.1, normalize=True):
        super(PixelSimLoss, self).__init__()
        self.sample_ratio = sample_ratio
        self.normalize = normalize

    def forward(self, featmap_src, featmap_tgt, visualize=False):
        B, C, H, W = featmap_src.shape
        sample_idx = [np.random.choice(H * W, int(H * W * self.sample_ratio), replace=False) for _ in range(B)]
        sample_idx = np.stack(sample_idx, axis=0).reshape([B, 1, int(H * W * self.sample_ratio)]).repeat(C, axis=1)
        f_src, f_tgt = featmap_src.view([B, C, H * W]).gather(2, torch.from_numpy(sample_idx).to(featmap_src.device)), \
            featmap_tgt.view([B, C, H * W]).gather(2, torch.from_numpy(sample_idx).to(featmap_tgt.device))
        A_src, A_tgt = torch.bmm(f_src.permute([0, 2, 1]), f_src), torch.bmm(f_tgt.permute([0, 2, 1]), f_tgt)
        if self.normalize:
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss_semantic = torch.mean(torch.norm(A_src - A_tgt, dim=[1, 2]) ** 2 / sample_idx.shape[-1])
        else:
            # loss_semantic = torch.mean(torch.norm(A_src - A_tgt, dim=(1, 2)) / sample_idx.shape[-1])
            loss_semantic = F.binary_cross_entropy(A_src, A_tgt)
        if visualize:
            import matplotlib.pyplot as plt
            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # fig, ax = plt.subplots(figsize=(10, 2))
            # im = ax.imshow(f_src[0].detach().cpu().numpy(), cmap='GnBu', vmin=0, vmax=1)
            # divider = make_axes_locatable(ax)
            # cax = divider.new_horizontal(size="5%", pad=1, pack_start=True)
            # fig.add_axes(cax)
            # fig.colorbar(im, cax=cax, orientation="vertical")
            # plt.show()

            fig, ax = plt.subplots(figsize=(10, 2))
            ax.imshow(f_src[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('f_src.png')
            plt.show()
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.imshow(f_tgt[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('f_tgt.png')
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(A_src[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('A_src.png')
            plt.show()
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(A_tgt[0].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            plt.savefig('A_tgt.png')
            plt.show()

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
    feat1, feat2 = torch.ones([B, C, H, W]) / C, torch.zeros([B, C, H, W])
    feat2[:, 1] = 1
    batch_loss = BatchSimLoss()
    l1 = batch_loss(feat1, feat2)
    pixel_loss_1 = PixelSimLoss(normalize=False)
    l2_1 = pixel_loss_1(feat1, feat2)
    pixel_loss_2 = PixelSimLoss()
    l2_2 = pixel_loss_2(feat1, feat2)
    channel_loss = ChannelSimLoss()
    l3 = channel_loss(feat1, feat2)
    feat1, feat2 = torch.randn([B, C]), torch.randn([B, C])
    l3_1 = channel_loss(feat1, feat2)
    pass
