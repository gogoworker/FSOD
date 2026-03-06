import torch
import torch.nn as nn

"""《Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection》AAAI2025
近年来，基于卷积神经网络 （CNN） 的红外小目标检测方法取得了出色的性能。然而，这些方法通常采用标准卷积，而忽略了考虑红外小目标像素分布的空间特性。
因此，我们提出了一种新的风车形卷积 （PConv） 来替代骨干网络下层的标准卷积。
PConv 更好地与暗淡小目标的像素高斯空间分布对齐，增强了特征提取，显著增加了感受野，并且仅引入了最小的参数增加。
此外，虽然最近的损失函数结合了尺度和位置损失，但它们没有充分考虑这些损失在不同目标尺度上的不同灵敏度，从而限制了对微小目标的检测性能。
为了克服这个问题，我们提出了一种基于尺度的动态 （SD） 损失，它根据目标大小动态调整尺度和位置损失的影响，从而提高网络检测不同尺度目标的能力。
我们构建了一个新的基准 SIRST-UAVB，这是迄今为止最大、最具挑战性的实拍单帧红外小目标检测数据集。
最后，通过将 PConv 和 SD Loss 集成到最新的小目标检测算法中，我们在 IRSTD-1K 和 SIRST-UAVB 数据集上实现了显著的性能改进，
验证了我们方法的有效性和通用性。
"""

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Pinwheel_shaped_Convolution(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

if __name__ == '__main__':
    block = Pinwheel_shaped_Convolution(c1=32, c2=32, k=3, s=1)  # c2要是4的倍数

    input = torch.rand(16, 32, 64, 64)

    output = block(input)

    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")
