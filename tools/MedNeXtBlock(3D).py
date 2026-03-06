import torch
import torch.nn as nn
import torch.nn.functional as F


"""《MedNeXt: Transformer-Driven Scaling of ConvNets for Medical Image Segmentation》 MICCAI 2023 
人们对采用基于 Transformer 的架构进行医学图像分割的兴趣呈爆炸式增长。然而，由于缺乏大规模的带注释的医学数据集，因此很难实现与自然图像相当的性能。相比之下，卷积网络具有更高的归纳偏差，因此很容易训练到高性能。
最近，ConvNeXt 架构试图通过镜像 Transformer 模块来实现标准 ConvNet 的现代化。在这项工作中，我们在此基础上进行了改进，设计了一个现代化且可扩展的卷积架构，针对数据稀缺医疗环境的挑战进行了定制。
我们介绍了 MedNeXt，这是一个受 Transformer 启发的大型内核分割网络，它引入了 – 
1） 用于医学图像分割的完全 ConvNeXt 3D 编码器-解码器网络，
2） 残余 ConvNeXt 上采样和下采样模块，以保持跨尺度的语义丰富性，
3） 一种新技术，通过上采样小型内核网络迭代增加内核大小，以防止有限医疗数据的性能饱和， 
4） MedNeXt 多个级别（深度、宽度、内核大小）的复合缩放。这导致 CT 和 MRI 模态的 4 项任务以及不同的数据集大小具有最先进的性能，代表了医学图像分割的现代化深度架构。
我们的代码在以下网址公开提供：https://github.com/MIC-DKFZ/MedNeXt。
"""


class MedNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 dim='3d',
                 grn=False
                 ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1



if __name__ == '__main__':
    block = MedNeXtBlock(in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group')
    input = torch.rand(2, 12, 64, 64, 64)
    output = block(input)
    print(input.size())
    print(output.size())
