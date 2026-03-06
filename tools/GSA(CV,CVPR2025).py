import torch
import torch.nn as nn


"""《DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation》 CVPR 2025
场景理解的最新进展得益于深度图，因为它包含三维几何信息，尤其是在复杂条件下（例如弱光和过度曝光）。现有方法将深度图与 RGB 图像一起编码，并在它们之间进行特征融合，以实现更稳健的预测。
考虑到深度可以被视为 RGB 图像的几何补充，一个简单的问题出现了：我们真的需要像对 RGB 图像那样，用神经网络显式地编码深度信息吗？基于这一见解，本文研究了一种学习 RGBD 特征表示的新方法，并提出了 DFormerv2
这是一个强大的 RGBD 编码器，它显式地使用深度图作为几何先验，而不是用神经网络编码深度信息。我们的目标是从所有图像块标记之间的深度和空间距离中提取几何线索，然后将其用作几何先验，在自注意力机制中分配注意力权重。
大量实验表明，DFormerv2 在各种 RGBD 语义分割基准测试中表现出色。代码可在以下位置获取：https://github.com/VCIP-RGBD/DFormer。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GSA(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos=None, split_or_not=False):
        """
        x: (b h w c)
        rel_pos: ((sin, cos), mask) 或 None
        """
        bsz, h, w, _ = x.size()

        # 创建默认的相对位置编码
        device = x.device
        # 创建默认的sin/cos (这里简化为全1)
        sin = cos = torch.ones((1, 1, 1, 1), device=device)
        # 创建默认的mask (这里简化为全0)
        mask = torch.zeros((self.num_heads, h * w, h * w), device=device)

        # 如果提供了rel_pos，
        if rel_pos is not None:
            (sin, cos), mask = rel_pos
            assert h * w == mask.size(1), f"mask size {mask.size()} doesn't match input size {h}x{w}"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = vr.flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


if __name__ == '__main__':
    embed_dim = 256
    num_heads = 8
    batch_size = 2
    height = width = 16

    block = GSA(embed_dim, num_heads).to('cuda')

    #  (batch_size, height, width, channel)
    input = torch.rand(batch_size, height, width, embed_dim).to('cuda')

    # 运行测试 - 现在不需要显式提供rel_pos参数
    output = block(input)

    # 打印结果
    print("输入尺寸:", input.size())
    print("输出尺寸:", output.size())