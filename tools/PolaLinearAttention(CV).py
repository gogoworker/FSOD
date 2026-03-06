import torch
import torch.nn as nn

"""《PolaFormer: Polarity-aware Linear Attention for Vision Transformers》ICLR 2025
线性注意力机制已成为基于 softmax 的注意力机制的一种有前途的替代方案，它利用核化特征图将序列长度的复杂性从二次函数降低到线性函数。
然而，与原始查询键点积相比，特征图上的非负约束和近似中使用的宽松指数函数会导致大量信息丢失，从而导致熵值较高、区分度较低的注意力图。
为了解决查询键对中负值驱动的缺失交互，我们提出了一种极性感知线性注意力机制（polarity-aware linear attention mechanism），
该机制明确模拟同符号和异符号查询键交互，确保全面覆盖关系信息。
此外，为了恢复注意力图的尖峰特性，我们提供了理论分析，证明存在一类元素函数（具有正的一阶和二阶导数），可以降低注意力分布中的熵。
为简单起见，并认识到每个维度的不同贡献，我们使用可学习的幂函数进行重新缩放，从而可以有效分离强和弱注意力信号。
大量实验表明，所提出的 PolaFormer 提高了各种视觉任务的性能，表现力和效率提高了 4.6%。
"""


class PolaLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, alpha, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)  # 生成查询 q 和 g，并将其拆开

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)  # 使用 sigmoid 来调整幂

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power  # 正符号查询
        q_neg = kernel_function(-q) ** power  # 负符号查询
        k_pos = kernel_function(k) ** power  # 正符号键
        k_neg = kernel_function(-k) ** power  # 负符号键

        # 合并正负符号的查询和键
        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)  # 对值进行切分，分为两部分

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z  # 正符号的相似性
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z  # 负符号的相似性

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N,
                                          mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x



if __name__ == '__main__':
    B = 2
    N = 64
    C = 128
    H = 8
    W = 8

    block = PolaLinearAttention(dim=C, num_patches=N, sr_ratio=1)
    input_tensor = torch.rand(B, N, C)

    output = block(input_tensor, H, W)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())
