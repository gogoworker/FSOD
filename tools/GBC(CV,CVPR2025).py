import torch
from torch import nn as nn

"""《SCSegamba: Lightweight Structure-Aware Vision Mamba for CrackSegmentation in Structures》 CVPR 2025
在各种场景中对结构裂缝进行像素级分割仍然是一项艰巨的挑战。当前的方法在有效建模裂缝形态和纹理方面遇到挑战，面临着在分割质量和低计算资源使用率之间取得平衡的挑战。
为了克服这些限制，我们提出了一种轻量级结构感知视觉曼巴网络 (SCSegamba)，能够通过利用裂缝像素的形态信息和纹理线索以最小的计算成本生成高质量的像素级分割图。
具体来说，我们开发了一个结构感知视觉状态空间模块 (SAVSS)，它结合了轻量级门控瓶颈卷积 (GBC) 和结构感知扫描策略 (SASS)。
GBC 的关键见解在于它能够有效地建模裂缝的形态信息，而 SASS 通过加强裂缝像素之间语义信息的连续性来增强对裂缝拓扑和纹理的感知。
在裂缝基准数据集上的实验表明，我们的方法优于其他最先进的 (SOTA) 方法，仅用 2.8M 个参数就实现了最高性能。
在多场景数据集上，我们的方法 F1 得分达到 0.8390，mIoU 达到 0.8479。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()

        # Ensure mid_channels is at least 2
        mid_channels = max(mid_channels, 2)

        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels,
                                   bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN') -> None:
        super().__init__()

        self.proj = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        if norm_type == 'GN':
            # Ensure num_groups is at least 1
            num_groups = max(in_channels // 16, 1)
            self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups)
        self.nonliner = nn.ReLU()

        self.proj2 = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        if norm_type == 'GN':
            num_groups2 = max(in_channels // 16, 1)
            self.norm2 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups2)
        self.nonliner2 = nn.ReLU()

        self.proj3 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        if norm_type == 'GN':
            num_groups3 = max(in_channels // 16, 1)
            self.norm3 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups3)
        self.nonliner3 = nn.ReLU()

        self.proj4 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        if norm_type == 'GN':
            num_groups4 = max(in_channels // 16, 1)
            self.norm4 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups4)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1_1 = self.proj(x)
        x1_1 = self.norm(x1_1)
        x1_1 = self.nonliner(x1_1)

        x1 = self.proj2(x1_1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 * x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual



if __name__ == '__main__':
    block = GBC(in_channels=3).to('cuda')

    input_tensor = torch.rand(1, 3, 32, 32).to('cuda')

    output_tensor = block(input_tensor)


    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output_tensor.size()}')