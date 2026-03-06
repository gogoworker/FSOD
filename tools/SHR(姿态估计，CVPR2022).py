import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange


"""《MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation》 CVPR 2022
由于深度模糊和自遮挡，从单目视频估计 3D 人体姿势是一项具有挑战性的任务。大多数现有研究试图通过利用空间和时间关系来解决这两个问题。然而，这些研究忽略了一个事实，即这是一个逆问题，其中存在多个可行解（即假设）。
为了缓解这一限制，我们提出了一个多假设变换器 (MHFormer)，它可以学习多个合理的姿势假设的时空表示。
为了有效地模拟多假设依赖关系并在假设特征之间建立强关系，该任务分为三个阶段：
（i）生成多个初始假设表示；
（ii）模拟自我假设通信，将多个假设合并为一个融合表示，然后将其划分为几个发散假设；
（iii）学习跨假设通信并聚合多假设特征以合成最终的 3D 姿势。
通过上述处理，最终的表征得到了增强，合成的姿势更加准确。大量实验表明，MHFormer 在两个具有挑战性的数据集 Human3.6M 和 MPI-INF-3DHP 上取得了最佳结果。
在没有任何花哨的修饰下，其性能在 Human3.6M 上以 3% 的大幅优势超越了之前的最佳结果。代码和模型可在 https://github.com/Vegetebird/MHFormer 上找到。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SHR_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.norm1_3 = norm_layer(dim)

        self.attn_1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> (b f) j c')

        x_1 = x
        x_2 = x
        x_3 = x

        x_1 = x_1 + self.drop_path(self.attn_1(self.norm1_1(x_1)))
        x_2 = x_2 + self.drop_path(self.attn_2(self.norm1_2(x_2)))
        x_3 = x_3 + self.drop_path(self.attn_3(self.norm1_3(x_3)))

        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2:]

        x = x_1 + x_2 + x_3

        x = rearrange(x, '(b f) j c -> b f j c', b=B, f=F)

        return x


if __name__ == '__main__':
    block = SHR_Block(dim=64, num_heads=8, mlp_hidden_dim=256).to('cuda')
    input = torch.rand(2, 16, 17, 64).to('cuda')
    output = block(input)
    print(input.size())
    print(output.size())