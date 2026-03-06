import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

torch.set_printoptions(threshold=np.inf)

"""《GRFormer: Grouped Residual Self-Attention for Lightweight Single Image Super-Resolution》ACM MM2024
先前的研究表明，减少基于 Transformer 的单图像超分辨率 (SISR) 模型（例如 SwinIR）的参数开销和计算量通常会导致性能下降。
在本文中，我们提出了一种高效、轻量级的方法 GRFormer，它不仅可以减少参数开销和计算量，而且可以大大提高性能。GRFormer 的核心是分组残差自注意力 (Grouped Residual Self-Attention, GRSA)，它专门面向两个基本组件。
首先，它引入了一个新颖的分组残差层 (GRL) 来替换自注意力中的查询、键、值 (QKV) 线性层，旨在同时有效地减少参数开销、计算量和性能损失。
其次，它集成了一个紧凑的指数空间相对位置偏差 (ES-RPB) 来替代原始的相对位置偏差，以提高表示位置信息的能力，同时进一步最小化参数数量。
大量实验结果表明，GRFormer 在 ×2、×3 和 ×4 SISR 任务中的表现均优于目前最先进的基于 Transformer 的方法，尤其是在 DIV2K 数据集上训练时，其峰值信噪比 (PSNR) 最高可比 SOTA 高出 0.23dB，
同时仅在自注意力模块中，参数数量和 MAC 数量分别减少了约60%和49%。我们希望我们简单有效的方法能够轻松应用于基于窗口划分自注意力的 SR 模型，并成为进一步研究图像超分辨率的有用工具。
"""
class GRSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim**-0.5

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # mlp to generate continuous relative position bias
        self.ESRPB_MLP = nn.Sequential(nn.Linear(2, 128, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, num_heads, bias=False))
        # 生成相对坐标表（relative_coords_table）
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_position_bias_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_position_bias_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_position_bias_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_position_bias_table *= 3.2  # normalize to -3.2, 3.2
        relative_position_bias_table = torch.sign(relative_position_bias_table) * (1 - torch.exp(
            -torch.abs(relative_position_bias_table)))
        self.register_buffer("relative_position_bias_table", relative_position_bias_table)

        # get pair-wise aligned relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)
        self.q1, self.q2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.k1, self.k2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.v1, self.v2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1, self.proj2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        x = x.reshape(x.shape[0], x.shape[1], 2, c // 2).permute(2, 0, 1, 3).contiguous()

        # GRL_k
        k = torch.stack((x[0] + self.k1(x[0]), x[1] + self.k2(x[1])), dim=0)
        k = k.permute(1, 2, 0, 3).flatten(2)
        k = k.reshape(b_, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # GRL_q
        q = torch.stack((x[0] + self.q1(x[0]), x[1] + self.q2(x[1])), dim=0)
        q = q.permute(1, 2, 0, 3).flatten(2)
        q = q.reshape(b_, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # GRL_v
        v = torch.stack((x[0] + self.v1(x[0]), x[1] + self.v2(x[1])), dim=0)
        v = v.permute(1, 2, 0, 3).flatten(2)
        v = v.reshape(b_, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        max_logit = torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))
        logit_scale = torch.clamp(self.logit_scale, max=max_logit).exp()
        attn = attn * logit_scale

        # 从MLP获取相对位置偏置
        relative_position_bias_table = self.ESRPB_MLP(self.relative_position_bias_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            n, n, -1)  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)  # 缩放偏置范围

        # 将ES-RPB加到注意力权重上
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = x.reshape(b_, n, 2, c // 2).permute(2, 0, 1, 3).contiguous()
        x = torch.stack((self.proj1(x[0]), self.proj2(x[1])), dim=0).permute(1, 2, 0, 3).reshape(b_, n, c)
        return x


if __name__ == '__main__':
    batch_size = 4
    num_patches = 16  # 假设是4x4的patch
    embed_dim = 64

    grsa = GRSA(dim=embed_dim, window_size=(4, 4), num_heads=4).to('cuda')

    input_tensor = torch.randn(batch_size, num_patches, embed_dim).to('cuda')

    output = grsa(input_tensor)

    print(input_tensor.size())
    print(output.size())
