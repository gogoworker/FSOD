import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

"""《3D-RCNet: Learning from Transformer to Build a 3D Relational ConvNet for Hyperspectral Image Classification》arxiv2024
最近，视觉变换器 (ViT) 模型以其优异的性能在各种计算机视觉任务中取代了经典的卷积神经网络 (ConvNet)。即使在高光谱图像 (HSI) 分类领域，基于 ViT 的方法也显示出巨大的潜力。
然而，ViT 在处理 HSI 数据时遇到了明显的困难。它的自注意力机制表现出二次复杂度，增加了计算成本。此外，ViT 对训练样本的巨大需求与 HSI 数据昂贵标记所带来的实际限制不符。
为了克服这些挑战，我们提出了一种名为 3D-RCNet 的 3D 关系卷积网络，它继承了 ConvNet 和 ViT 的优势，在 HSI 分类中表现出色。我们将 Transformer 的自注意力机制嵌入到 ConvNet 的卷积运算中，
设计了 3D 关系卷积运算并使用它来构建最终的 3D-RCNet。所提出的 3D-RCNet 既保留了 ConvNet 的高计算效率，又具有 ViT 的灵活性。
此外，所提出的 3D 关系卷积运算是一种即插即用运算，可以无缝插入到以前基于 ConvNet 的 HSI 分类方法中。在三个代表性基准 HSI 数据集上进行的经验评估表明，所提出的模型优于以前基于 ConvNet 和基于 ViT 的 HSI 方法。
"""


class unfold_3d(nn.Module):
    def __init__(self, kernel_size, stride, padd=[1, 1, 1], padd_mode='replicate'):
        super(unfold_3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padd = padd
        self.padd_mode = padd_mode

    def forward(self, x):
        x = F.pad(x, (self.padd[0], self.padd[0], self.padd[1], self.padd[1], self.padd[2], self.padd[2]),
                  mode=self.padd_mode)
        # x = F.ReflectionPad3d(x, [0, 0, padd[0], padd[1], padd[2]], mode=padd_mode)
        x = x.unfold(2, self.kernel_size[0], self.stride[0]) \
            .unfold(3, self.kernel_size[1], self.stride[1]) \
            .unfold(4, self.kernel_size[2], self.stride[2])
        # x = rearrange(x, 'b c h w d k1 k2 k3 -> b h w d (k1 k2 k3) c')
        x = rearrange(x, 'b c h w d k1 k2 k3 -> b (h w d) (k1 k2 k3) c')
        return x


class MLP_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim * 3),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim * 3, dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        return self.net(x)


class Rconv_3D(nn.Module):
    def __init__(self, dim, kernel_size=[3, 3, 3], stride=[1, 1, 1], heads=4):
        super(Rconv_3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padd = [kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2]
        self.num_heads = heads

        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=1),
            nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        )

        self.norm = nn.Sequential(
            Rearrange('b c h w d -> b (h w d) c'),
            # nn.LayerNorm(dim)
        )

        self.unfold_k = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )
        self.unfold_v = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )

    def forward(self, x):
        B, C, H, W, S = x.shape

        qkv = self.qkv(x).reshape(B, 3, C, H, W, S)
        q, k, v = qkv.unbind(1)
        q = self.norm(q)
        k = self.unfold_k(k)
        v = self.unfold_v(v)

        B, L, K, C = k.shape
        # q = q.reshape(B, self.num_heads, L, 1, -1)
        q = q.contiguous().view(B, self.num_heads, L, 1, -1)  # (B,head,(h*w*d),1,c/head)
        k = k.view(B, self.num_heads, L, K, -1)  # (B,head,(hwd),27,c/head)
        v = v.view(B, self.num_heads, L, K, -1)

        attn = q @ k.transpose(-2, -1)  # (B,head,(hwd),1,27)
        attn = (attn * self.scale).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)  # (B,head,(hwd),1,c/head)
        # x = torch.einsum('bhlxk,bhlkc->bhlxc', attn, v)
        x = x.reshape(B, L, C).transpose(-2, -1).reshape(B, C, H, W, S)  # B, n, C
        x = self.proj(x)
        return x

if __name__ == '__main__':
    input_tensor = torch.rand(1, 16, 8, 8, 8)

    block = Rconv_3D(dim=16)

    output = block(input_tensor)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())