import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
from einops import rearrange


"""《SEM-Net: Efficient Pixel Modelling for image inpainting with Spatially Enhanced SSM》 WACV 2025
图像修复旨在基于图像已知区域的信息修复部分受损图像。实现语义上合理的修复结果尤其具有挑战性，因为它要求重建区域与语义上一致的区域呈现出相似的模式。这需要一个能够捕捉长距离依赖关系的模型。
现有模型在这方面表现不佳，原因是基于卷积神经网络 (CNN) 的方法的感受野增长缓慢，以及基于 Transformer 的方法中块级交互无法有效捕捉长距离依赖关系。
受此启发，我们提出了 SEM-Net，一种新颖的视觉状态空间模型 (SSM) 视觉网络，它在像素级对损坏图像进行建模，同时在状态空间中捕捉长距离依赖关系 (LRD)，并实现线性计算复杂度。
为了解决 SSM 固有的空间感知不足问题，我们引入了 Snake Mamba Block (SMB) 和空间增强前馈网络(Spatially-Enhanced Feedforward Network,SEFN)。
这些创新使 SEM-Net 在两个不同的数据集上超越了最先进的修复方法，在捕捉长距离相关点 (LRD) 方面表现出显著的改进，并增强了空间一致性。
此外，SEM-Net 在运动去模糊方面也达到了最先进的性能，证明了其通用性。
"""



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SEFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SEFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                            groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, spatial):
        x = self.project_in(x)

        #### Spatial branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)
        ####

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


if __name__ == '__main__':
    dim = 64
    ffn_expansion_factor = 2  # 隐藏层扩展系数
    bias = True  # 是否使用偏置

    block = SEFN(dim=dim,
                 ffn_expansion_factor=ffn_expansion_factor,
                 bias=bias).to('cuda')

    batch_size = 4
    height, width = 64, 64

    # 主输入x和空间信息输入spatial
    x = torch.rand(batch_size, dim, height, width).to('cuda')
    spatial = torch.rand(batch_size, dim, height, width).to('cuda')

    output = block(x, spatial)


    print("主输入x尺寸:", x.size())
    print("空间输入spatial尺寸:", spatial.size())
    print("输出output尺寸:", output.size())