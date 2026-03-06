import torch

"""《SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design》CVPR2024
最近，高效的视觉Transformer在资源受限设备上展示了出色的性能和低延迟。传统上，这些模型在宏观层面使用4×4的patch嵌入和4阶段结构，而在微观层面利用复杂的多头注意力机制。
本研究旨在以内存高效的方式解决所有设计层面的计算冗余问题。我们发现，使用较大步幅的patchify stem不仅可以降低内存访问成本，还能通过减少早期阶段的空间冗余token表示，达到具有竞争力的性能。
此外，我们的初步分析表明，早期阶段的注意力层可以用卷积代替，而后期阶段的几个注意力头存在计算冗余。
为了解决这个问题，我们引入了一个单头注意力模块，该模块能够本质上避免头部冗余，并通过并行结合全局和局部信息同时提升精度。
基于这些解决方案，我们提出了SHViT（Single-Head Vision Transformer），一种实现速度与精度最佳平衡的单头视觉Transformer。
例如，在ImageNet-1k数据集上，我们的SHViT-S4在GPU、CPU和iPhone12移动设备上的速度分别比MobileViTv2×1.0快3.3倍、8.1倍和2.4倍，同时准确率提高了1.3%。
在MS COCO数据集上进行目标检测和实例分割时，使用Mask R-CNN head，我们的模型性能与FastViT-SA12相当，但在GPU和移动设备上的主干延迟分别降低了3.8倍和2.0倍。
"""

class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""

    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))

        return x

if __name__ == '__main__':
    block = SHSA(dim=64, qk_dim=16, pdim=32).to('cuda')

    input_tensor = torch.rand(2, 64, 32, 32).to('cuda')

    output_tensor = block(input_tensor)

    print(f"Input size: {input_tensor.size()}")
    print(f"Output size: {output_tensor.size()}")