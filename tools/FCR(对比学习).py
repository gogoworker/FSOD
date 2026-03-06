import torch
import torch.nn as nn
from numpy import *
import random

"""《Efficient Frequency-Domain Image Deraining with Contrastive Regularization》ECCV 2024
目前大多数单幅图像去雨 (SID) 方法都是基于 Transformer 进行全局建模以实现高质量重建。然而，它们的架构仅从空间域构建长距离特征，这承受着巨大的计算负担以保持有效性。
此外，这些方法要么在训练中忽略了负样本信息，要么没有充分利用负样本中存在的雨纹图案。为了解决这些问题，我们提出了一个频率感知去雨 Transformer 框架 (FADformer)，它可以完全捕获频域特征以有效去除雨水。
具体来说，我们构建了 FADBlock，包括融合傅里叶卷积混合器 (FFCM) 和先验门控前馈网络 (PGFN)。与自注意力机制不同，FFCM 在空间和频域进行卷积运算，使其具有局部全局捕获能力和效率。
同时，PGFN 以门控方式引入残差通道先验，以增强局部细节并保留特征结构。
此外，我们在训练过程中引入了频域对比正则化 (FCR)。FCR 促进了频域中的对比学习，并利用负样本中的雨条模式来提高性能。
"""
# 箫张跋扈整理：https://space.bilibili.com/478113245

def sample_with_j(k, n, j):
    if n >= k:
        raise ValueError("n must be less than k.")
    if j < 0 or j > k:
        raise ValueError("j must be in the range 0 to k.")

    # 创建包含0到k的数字的列表
    numbers = list(range(k))

    # 确保j在数字列表中
    if j not in numbers:
        raise ValueError("j must be in the range 0 to k.")

    # 从数字列表中选择j
    sample = [j]

    # 从剩余的数字中选择n-1个
    remaining = [num for num in numbers if num != j]
    sample.extend(random.sample(remaining, n - 1))

    return sample


# -------------------FCR----------------------- #
# Frequency Contrastive Regularization

class FCR(nn.Module):
    def __init__(self, ablation=False):

        super(FCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        contrastive = 0
        for i in range(a_fft.shape[0]):
            d_ap = self.l1(a_fft[i], p_fft[i])
            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):
                d_an = self.l1(a_fft[i], n_fft[j])
                contrastive += (d_ap / (d_an + 1e-7))
        contrastive = contrastive / (self.multi_n_num * a_fft.shape[0])

        return contrastive

if __name__ == '__main__':
    block = FCR().to('cuda')

    batch_size = 4
    channels = 3
    height = 32
    width = 32

    # 锚点样本 (a)
    a = torch.rand(batch_size, channels, height, width).to('cuda')
    # 正样本 (p)
    p = torch.rand(batch_size, channels, height, width).to('cuda')
    # 负样本 (n)
    n = torch.rand(batch_size, channels, height, width).to('cuda')

    output = block(a, p, n)

    print("锚点样本 (a) 的形状:", a.size())
    print("正样本 (p) 的形状:", p.size())
    print("负样本 (n) 的形状:", n.size())
    print("对比损失值:", output.item())
