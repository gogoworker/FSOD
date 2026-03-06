from torch.autograd import Function
import torch
import torch.nn as nn


"""《High-Similarity-Pass Attention for Single Image Super-Resolution》   IEEE TIP 2024（中科院一区TOP）
非局部注意力 (NLA) 领域的最新发展引发了人们对基于自相似性的单图像超分辨率 (SISR) 的新兴趣。研究人员通常使用 NLA 来探索 SISR 中的非局部自相似性 (NSS) 并获得令人满意的重建结果。
然而，标准 NLA 的重建性能与具有随机选择区域的 NLA 的重建性能相似的令人惊讶的现象促使我们重新审视 NLA。
在本文中，我们首先从不同角度分析了标准 NLA 的注意力图，发现得到的概率分布总是对每个局部特征都有完全支持，这意味着将值分配给不相关的非局部特征是一种统计浪费，特别是对于需要使用大量冗余非局部特征来建模长程依赖性的 SISR。
基于这些发现，我们引入了一种简洁而有效的软阈值操作来获得高相似性传递注意力 (HSPA)，这有利于生成更紧凑和可解释的分布。
此外，我们推导出软阈值操作的一些关键属性，这些属性使我们能够以端到端的方式训练我们的 HSPA。
HSPA 可以作为高效的通用构建块集成到现有的深度 SISR 模型中。此外，为了证明 HSPA 的有效性，我们通过在简单的主干中集成几个 HSPA 构建了一个深度高相似度传递注意网络 (HSPAN)。
大量实验结果表明，HSPAN 在定量和定性评估方面均优于最先进的方法。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class SoftThresholdingOperation(nn.Module):
    def __init__(self, dim=2, topk=128):
        super(SoftThresholdingOperation, self).__init__()
        self.dim = dim
        self.topk = topk

    def forward(self, x):
        return softThresholdingOperation(x, self.dim, self.topk)

def softThresholdingOperation(x, dim=2, topk=128):
    return SoftThresholdingOperationFun.apply(x, dim, topk)

class SoftThresholdingOperationFun(Function):
    @classmethod
    def forward(cls, ctx, s, dim=2, topk=128):
        ctx.dim = dim
        max, _ = s.max(dim=dim, keepdim=True)
        s = s - max
        tau, supp_size = tau_support(s, dim=dim, topk=topk)
        output = torch.clamp(s - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output
    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None

def tau_support(s, dim=2, topk=128):
    if topk is None or topk >= s.shape[dim]:
        k, _ = torch.sort(s, dim=dim, descending=True)
    else:
        k, _ = torch.topk(s, k=topk, dim=dim)

    topk_cumsum = k.cumsum(dim) - 1
    ar_x = ix_like_fun(k, dim)
    support = ar_x * k > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(s.dtype)

    if topk is not None and topk < s.shape[dim]:
        unsolved = (support_size == topk).squeeze(dim)

        if torch.any(unsolved):
            in_1 = roll_fun(s, dim)[unsolved]
            tau_1, support_size_1 = tau_support(in_1, dim=-1, topk=2 * topk)
            roll_fun(tau, dim)[unsolved] = tau_1
            roll_fun(support_size, dim)[unsolved] = support_size_1

    return tau, support_size

def ix_like_fun(x, dim):
    d = x.size(dim)
    ar_x = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return ar_x.view(view).transpose(0, dim)

def roll_fun(x, dim):
    if dim == -1:
        return x
    elif dim < 0:
        dim = x.dim() - dim

    perm = [i for i in range(x.dim()) if i != dim] + [dim]
    return x.permute(perm)


# High-Similarity-Pass Attention
class HSPA(nn.Module):
    def __init__(self, channel=256, reduction=2, res_scale=1,conv=default_conv, topk=128):
        super(HSPA, self).__init__()
        self.res_scale = res_scale
        self.conv_match1 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())
        self.ST = SoftThresholdingOperation(dim=2, topk=topk)

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N,C,H,W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
        x_embed_2 = x_embed_2.view(N,C,H*W)

        score = torch.matmul(x_embed_1, x_embed_2)
        score = self.ST(score)

        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
        x_final = torch.matmul(score, x_assembly)
        return self.res_scale*x_final.permute(0,2,1).view(N,-1,H,W)+input


if __name__ == '__main__':
    input = torch.rand(2, 16, 8, 8).to('cuda')

    block = HSPA(channel=16).to('cuda')

    output = block(input)

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)