import torch
from torch import nn
from einops import rearrange

"""《Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few》 NeurIPS2025
Transformer 模型中的注意力机制已取得显著的实验成功。然而，其前向传播的优化目标仍然不明确。
此外，自注意力机制的二次复杂度也日益令人望而却步。与以往分别解决可解释性或效率问题的研究不同，我们提出了一种统一的优化目标，以同时缓解这两个问题。通过将优化过程展开到目标函数上，我们得到了一种本质上可解释且高效的注意力机制。
该机制通过收缩少数代表性词元，然后将收缩结果广播回去，从而将所有词元压缩成低维结构。这种收缩广播自注意力（CBSA）机制不仅可以线性扩展，还可以将现有的注意力机制泛化为自身的特例。
实验进一步表明，CBSA 在多个视觉任务上表现出可比的性能，甚至更胜一筹。
"""

class CBSA(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, inner_dim, bias=False)

        self.step_x = nn.Parameter(torch.randn(heads, 1, 1))
        self.step_rep = nn.Parameter(torch.randn(heads, 1, 1))

        self.to_out = nn.Linear(inner_dim, dim)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        self.qkv = nn.Identity()

    def attention(self, query, key, value):
        dots = (query @ key.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = attn @ value
        return out, attn

    def forward(self, x, return_attn=False):
        b, n, c = x.shape
        h = width = int(n ** 0.5)

        w = self.proj(x)

        rep = self.pool(
            w.reshape(b, h, width, -1)
            .permute(0, 3, 1, 2)
        ).reshape(b, -1, h*width)
        rep = rep.permute(0, 2, 1)

        w = w.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (b, heads, n, dim_head)
        rep = rep.reshape(b, 64, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (b, heads, 64, dim_head)

        rep_delta, attn = self.attention(rep, w, w)
        if return_attn:
            return attn.transpose(-1, -2) @ attn

        rep = rep + self.step_rep * rep_delta

        x_delta, _ = self.attention(rep, rep, rep)
        x_delta = attn.transpose(-1, -2) @ x_delta
        x_delta = self.step_x * x_delta

        x_delta = rearrange(x_delta, 'b h n k -> b n (h k)')  # 合并多头维度
        return self.to_out(x_delta)

if __name__ == '__main__':

    model = CBSA(dim=8, heads=4, dim_head=16).to('cuda')

    input_tensor = torch.randn(1, 64, 8).to('cuda')

    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")