import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""《ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement》AAAI2025
医学图像分割在临床决策、治疗计划和疾病跟踪中发挥着重要作用。然而，它仍然面临两大挑战。
一方面，医学图像中的前景和背景之间通常存在“软边界”，光照不足和对比度低进一步降低了图像中前景和背景的可区分性。
另一方面，共现现象在医学图像中普遍存在，学习这些特征会误导模型的判断。为了应对这些挑战，我们提出了一个称为对比度驱动的医学图像分割（ConDSeg）的通用框架。
首先，我们开发了一种称为一致性强化的对比训练策略。它旨在提高编码器在各种光照和对比度场景下的鲁棒性，使模型即使在恶劣的环境中也能提取高质量的特征。
其次，我们引入了一个语义信息解耦模块，它能够将特征从编码器解耦为前景、背景和不确定区域，逐渐获得在训练过程中减少不确定性的能力。
然后，对比驱动特征聚合模块（ContrastDriven Feature Aggregation (CDFA) ）对比前景和背景特征，以指导多级特征融合和关键特征增强，进一步区分要分割的实体。我们还提出了一个尺寸感知解码器来解决解码器的尺度奇异性。
它准确地定位图像中不同大小的实体，从而避免错误地学习共现特征。在三个场景的五个医学图像数据集上进行的大量实验证明了我们方法的领先性能，
证明了其先进性和对各种医学图像分割场景的普遍适用性。
"""

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5


        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted

if __name__ == '__main__':
    batch_size = 2
    in_channels = 64
    height = 128
    width = 128
    dim = 64
    num_heads = 8
    kernel_size = 3
    stride = 2
    padding = 1
    attn_drop = 0.1
    proj_drop = 0.1

    block = ContrastDrivenFeatureAggregation(in_c=in_channels, dim=dim, num_heads=num_heads,
                                             kernel_size=kernel_size, padding=padding, stride=stride,
                                             attn_drop=attn_drop, proj_drop=proj_drop)

    x = torch.rand(batch_size, in_channels, height, width)  # 输入图像
    fg = torch.rand(batch_size, in_channels, height, width)  # 前景特征图
    bg = torch.rand(batch_size, in_channels, height, width)  # 背景特征图

    output = block(x, fg, bg)

    print(f"Input size (x): {x.size()}")
    print(f"Input size (fg): {fg.size()}")
    print(f"Input size (bg): {bg.size()}")
    print(f"Output size: {output.size()}")