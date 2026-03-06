import torch
import torch.nn as nn
import torch.nn.functional as F

"""ICCV 2023 《Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution》
尽管基于深度学习的解决方案在图像超分辨率 (SR) 中实现了令人印象深刻的重建性能，但这些模型通常很大，具有复杂的架构，使其与具有许多计算和内存限制的低功耗设备不兼容。
为了克服这些挑战，我们提出了一种空间自适应特征调制 (SAFM) 机制来实现高效的 SR 设计。具体来说，SAFM 层使用独立计算来学习多尺度特征表示并聚合这些特征以进行动态空间调制。
由于 SAFM 优先利用非局部特征依赖性，我们进一步引入了卷积通道混合器 (CCM) 来编码局部上下文信息并同时混合通道。
大量的实验结果表明，所提出的方法比最先进的高效 SR 方法（例如 IMDN）小 3 倍，并且以更少的内存使用量获得相当的性能。我们的源代码和预训练模型可在以下位置获得：https://github.com/sunny2109/SAFMN。
"""

# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

if __name__ == '__main__':
    batch_size = 2
    channels = 16
    height, width = 32, 32

    dim = channels
    n_levels = 4
    block = SAFM(dim=dim, n_levels=n_levels)

    input = torch.rand(batch_size, dim, height, width)

    output = block(input)

    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")