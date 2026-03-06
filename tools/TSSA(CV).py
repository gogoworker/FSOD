import torch
import torch.nn as nn
from einops import rearrange
from thop import profile

"""《Token Statistics Transformer: Linear-Time Attention via Variational Rate Reduction》ICLR2025
注意力操作符可以说是 Transformer 架构的关键区别因素，它在各种任务上都表现出了最先进的性能。然而，Transformer 注意力操作符通常会带来很大的计算负担，计算复杂度与 token 数量成二次方关系。
在这项工作中，我们提出了一种新颖的 Transformer 注意力操作符，其计算复杂度与 token 数量成线性关系。
我们通过扩展先前的工作来推导出我们的网络架构，该工作表明，Transformer 风格的架构自然地由“白盒”架构设计产生，其中网络的每一层都设计为实现最大编码率降低目标 (MCR) 的增量优化步骤.
具体来说，我们推导出了 MCR 的一个新变分形式目标，并表明由该变分目标的展开梯度下降产生的架构导致了一个名为 Token Statistics Self-Attention (TSSA) 的新注意力模块。
TSSA 具有线性计算和内存复杂度，并且与计算 token 之间成对相似性的典型注意力架构截然不同。
在视觉、语言和长序列任务上的实验表明，只需将 TSSA 换成标准自注意力（我们称之为 Token Statistics Transformer (ToST)），就可以实现与传统 Transformer 相媲美的性能，同时计算效率和可解释性都更高。
我们的结果也在一定程度上质疑了成对相似性风格注意力机制对于 Transformer 架构成功至关重要的传统观点。代码将在 https://github.com/RobinWu218/ToST 提供。
"""

class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temp'}

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    batch_size = 32
    seq_length = 784
    embedding_dim = 128
    input = torch.rand(batch_size, seq_length, embedding_dim).to('cuda')

    block = AttentionTSSA(dim=embedding_dim).to('cuda')
    block1 = Attention(dim=embedding_dim).to('cuda')

    output = block(input)

    print(input.size())
    print(output.size())


    flops, params = profile(block, (input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    flops1, params1 = profile(block1, (input,))
    print('FLOPs1 = ' + str(flops1 / 1000 ** 3) + 'G')
    print('Params1 = ' + str(params1 / 1000 ** 2) + 'M')