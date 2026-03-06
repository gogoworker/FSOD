import torch.nn as nn
import torch
import math
from timm.models.layers import trunc_normal_


"""《MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation》 WACV 2024
除了 Transformer 之外，探索如何充分利用 MetaFormer 的容量也至关重要，MetaFormer 架构是 Transformer 性能提升的基础。
先前的研究仅将其用于主干网络。与先前的研究不同，我们更深入地探索了 MetaFormer 架构在语义分割任务中的容量。我们提出了一个强大的语义分割网络 MetaSeg，它将 Metaformer 架构从主干网络一直延伸到解码器。
我们的 MetaSeg 表明，MetaFormer 架构在为解码器和主干网络捕获有用的上下文信息方面发挥着重要作用。
此外，最近的分割方法表明，使用基于 CNN 的主干网络提取空间信息，并使用解码器提取全局信息，比使用基于 Transformer 的主干网络和基于 CNN 的解码器更有效。这促使我们采用基于 CNN 的主干网络，
并使用 MetaFormer 模块，并设计基于 MetaFormer 的解码器，该解码器包含一个新颖的自注意力模块来捕捉全局上下文信息。
为了兼顾全局上下文提取和语义分割中自注意力机制的计算效率，我们提出了一个通道缩减注意力 (CRA) 模块，将查询和键的通道维度缩减为一维。
如此一来，我们提出的 MetaSeg 在主流语义分割和医学图像分割基准测试（包括 ADE20K、Cityscapes、COCO-stuff 和 Synapse）上，以更高效的计算成本超越了之前的先进方法。
"""


class ChannelReductionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        # self.dim2 = dim2
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)  
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)  
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)  
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        B, N, C = x.shape  
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    input_tensor = torch.randn(4, 256, 8).to('cuda')

    block = ChannelReductionAttention(dim1=8).to('cuda')

    output = block(input_tensor, 16, 16)

    print("输入尺寸:", input_tensor.size())
    print("输出尺寸:", output.size())