import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
import torch
from torch import nn
import torch.nn.functional as F
from color_distillation.models.cyclegan import GeneratorResNet
from color_distillation.models.dncnn import DnCNN
from color_distillation.models.unet import UNet
from color_distillation.models.styleunet import StyleUNet
from color_distillation.models.unext import UNext
from torchvision.models.vision_transformer import MLPBlock, vit_b_16


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Parameter(torch.randn([self._num_embeddings, self._embedding_dim]))
        self._commitment_cost = commitment_cost

    def forward(self, inputs, num_colors):
        # convert inputs from BCHW -> BHWC
        B, C, H, W = inputs.shape

        # Calculate distances
        distances = torch.cdist(inputs.permute(0, 2, 3, 1).flatten(1, 2), self._embedding)
        # find small distance indices
        palette_idx = torch.topk(distances.norm(dim=1), num_colors, dim=1)[1]

        # Encoding
        encoding_idx = torch.argmin(torch.gather(distances, 2, palette_idx[:, None].repeat(1, H * W, 1)),
                                    dim=2, keepdim=True)
        encodings = torch.zeros([B, H * W, num_colors], device=inputs.device).scatter_(2, encoding_idx, 1)

        # Quantize and unflatten
        # M = encodings.view(B, H, W, N).permute(0, 3, 1, 2)
        # color_palette = (inputs[:, None] * M[:, :, None]).sum(dim=[3, 4]) / (M[:, :, None].sum(dim=[3, 4]) + 1e-8)
        # mean_quantized = (M[:, :, None] * color_palette[:, :, :, None, None]).sum(dim=1)
        emb_quantized = (encodings @ self._embedding[palette_idx]).view(B, H, W, C).permute(0, 3, 1, 2)
        color_palette = self._embedding[palette_idx]

        # Loss
        e_latent_loss = F.mse_loss(emb_quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(emb_quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        emb_quantized = inputs + (emb_quantized - inputs).detach()

        return emb_quantized, encodings.view(B, H, W, -1).permute(0, 3, 1, 2), color_palette, loss


class ColorEncoder(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int = 8,
            hidden_dim: int = 128,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.conv_in = nn.Sequential(nn.Conv2d(3, hidden_dim, 3, 2, 1),
                                     nn.BatchNorm2d(hidden_dim), nn.ReLU(),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                                     nn.BatchNorm2d(hidden_dim), nn.ReLU(), )

        self.depth_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.key_head = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, hidden_dim, dropout)

        # color output head
        self.color_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 3), nn.Sigmoid())

    # def forward(self, feat, color_query, num_colors):
    def forward(self, img, color_query, num_colors):
        feat = self.conv_in(img)
        feat = self.depth_conv(feat) + feat
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).permute(0, 2, 1)
        feat = self.ln_1(feat)
        k = self.key_head(feat)
        v = self.value_head(feat)
        # B, colors_channel, L
        m_all = (k @ color_query.T).permute(0, 2, 1)
        _, indices = torch.topk(m_all.norm(dim=-1), num_colors, 1)
        # B, num_colors, L
        m = torch.gather(m_all, 1, indices[:, :, None].repeat([1, 1, H * W]))
        x = F.softmax(m / C ** 0.5, dim=2) @ v.permute(0, 1, 2)
        x = self.output_head(x)
        # x, m = self.self_attention(query=color_query[None, :, :].repeat([B, 1, 1]), key=feat, value=feat)
        x = x.view([B, -1, num_colors, C]).mean(dim=1)
        m = torch.log(m).view([B, -1, num_colors, H * W]).mean(dim=1)
        x = self.dropout(x)

        y = self.ln_2(x)
        y = self.mlp(y)
        color_palette = self.color_head(x + y)
        return color_palette.permute(0, 2, 1), m.view([B, num_colors, H, W])


class ColorCNN(nn.Module):
    def __init__(self, arch, temp=1.0, bottleneck_channel=16, colors_channel=256, topk=4, agg='mean'):
        super().__init__()
        self.topk = topk
        self.temp = temp
        self.arch = arch
        if arch == 'unet':
            self.base = UNet(feature_dim=64)
        elif arch == 'unext':
            self.base = UNext()
        elif arch == 'dncnn':
            self.base = DnCNN()
        elif arch == 'cyclegan':
            self.base = GeneratorResNet()
        elif arch == 'styleunet':
            self.base = StyleUNet()
        else:
            raise Exception
        if bottleneck_channel != 0:
            self.bottleneck = nn.Sequential(nn.Conv2d(self.base.out_channel, bottleneck_channel, 1),
                                            nn.BatchNorm2d(bottleneck_channel), nn.ReLU(), )
        else:
            self.bottleneck = nn.Sequential()
            bottleneck_channel = self.base.out_channel
        # support color quantization into 256 colors at most
        # self.color_mask = nn.Parameter(torch.randn([bottleneck_channel, colors_channel]))
        self.color_mask = nn.Conv2d(bottleneck_channel, colors_channel, 1, bias=False)
        self.agg = agg

        # self.color_query = nn.Parameter(torch.randn([colors_channel, hidden_dim]))
        # self.color_encoder = ColorEncoder(8, hidden_dim)

        # self.color_layer = nn.Conv2d(bottleneck_channel, 3, 1)
        # self.vq_layer = VectorQuantizer(colors_channel, bottleneck_channel, 0.25)

    def forward(self, img, num_colors, mode='train'):
        B, _, H, W = img.shape
        if 'style' not in self.arch:
            feat = self.bottleneck(self.base(img))
        else:
            feat = self.bottleneck(self.base(img, int(np.log2(num_colors) - 1) * torch.ones([B]).long().to(img.device)))
        # m = (feat).view([B, -1, H * W]).permute([0, 2, 1]) @ (self.color_mask)
        # m = m.permute([0, 2, 1]).view([B, -1, H, W])
        m = self.color_mask(feat)
        if self.agg == 'mean':
            m = m.view([B, -1, num_colors, H, W]).mean(dim=1)
        elif self.agg == 'max':
            _, indices = torch.topk(m.view(B, -1, H * W).norm(dim=-1), num_colors, 1)
            # B, num_colors, L
            m = torch.gather(m, 1, indices[:, :, None, None].repeat([1, 1, H, W]))
            # m, _ = m.view([B, -1, num_colors, H, W]).max(dim=1)
        else:
            raise Exception
        # color_palette, _ = self.color_encoder(img, self.color_query, num_colors)
        # color_palette = color_palette[:, :, :, None, None]
        # m = F.softmax(m / self.temp, dim=1)
        topk, idx = torch.topk(m, min(self.topk, num_colors), dim=1)
        m = torch.scatter(torch.zeros_like(m), 1, idx, F.softmax(topk / self.temp, dim=1))  # softmax output
        color_palette = (img.unsqueeze(2) * m.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                m.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
        if mode == 'train':
            transformed_img = (m.unsqueeze(1) * color_palette).sum(dim=2)
        else:
            M = torch.argmax(m, dim=1, keepdim=True)  # argmax color index map
            M = torch.zeros_like(m).scatter(1, M, 1)
            transformed_img = (M.unsqueeze(1) * color_palette).sum(dim=2)

        return transformed_img, m, color_palette


if __name__ == '__main__':
    from color_distillation import datasets
    import color_distillation.utils.transforms as T
    from color_distillation.loss.similarity_preserving import PixelSimLoss

    # vit_model = vit_b_16()
    # _ = vit_model(torch.zeros([8, 3, 224, 224]))

    num_colors = 2
    # color_encoder = ColorEncoder(768)
    # color_query = nn.Parameter(torch.randn([256, 768]))
    # color_palette, m = color_encoder(torch.randn([8, 768, 32, 32]), color_query, num_colors)
    dataset = datasets.CIFAR10('/home/houyz/Data/cifar10', color_quantize=T.MedianCut())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=0, pin_memory=True)
    dataloader.dataset.num_colors[0] = num_colors
    img, target = next(iter(dataloader))
    img, label, quantized_img, index_map = img, target[0], target[1][0], target[1][1]
    B, C, H, W = img.shape
    model = ColorCNN('unext')
    trans_img, m, color_palette = model(img, num_colors, mode='train')
    M = torch.zeros_like(m).scatter(1, index_map, 1)
    loss_1 = PixelSimLoss(1, normalize=False)(m, M)
    loss_2 = PixelSimLoss(normalize=False)(m, M)
    loss_3 = PixelSimLoss(1, normalize=True)(m, M)
    loss_4 = PixelSimLoss(normalize=True)(m, M)
    loss_ce = F.cross_entropy(m, M.argmax(dim=1))
    crop = T.RandomCrop(32, padding=4)
    crop(trans_img)
    pass
