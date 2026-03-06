import torch
from torch import nn
import torch.nn.functional as F

"""《Efficient Face Super-Resolution via Wavelet-based Feature Enhancement Network》ACM2024
人脸超分辨率旨在从低分辨率人脸图像中重建高分辨率人脸图像。以前的方法通常采用编码器-解码器结构来提取面部结构特征，其中直接下采样不可避免地会引入失真，尤其是对于边缘等高频特征。
为了解决这个问题，我们提出了一种基于小波的特征增强网络，它通过使用小波变换将输入特征无损地分解为高频和低频分量并分别处理它们来减轻特征失真。
为了提高面部特征提取的效率，进一步提出了一种全域 Transformer 来增强局部、区域和全局的面部特征。这样的设计使我们的方法能够更好地执行，而无需像以前的方法那样堆叠许多模块。
实验表明，我们的方法有效地平衡了性能、模型大小和速度。代码链接：https://github.com/PRIS-CV/WFEN。
"""

class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h horizontal
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        #v vertical
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        #d diagonal
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


class WFD(nn.Module):
    def __init__(self, dim_in, dim, need=False):
        super(WFD, self).__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1, padding=0)
            self.HaarWavelet = HaarWavelet(dim, grad=False)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=False)
            self.dim = dim_in

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)

        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        return a+(h + v + d)


if __name__ == '__main__':
    dim_in = 64
    dim = 64

    input = torch.rand(8, dim_in, 32, 32)

    block = WFD(dim_in=dim_in, dim=dim, need=True)

    output = block(input)

    print("Input size:", input.size())
    # a, hvd = output
    # print("a size:", a.size())  # 输出 a 的尺寸
    # print("h + v + d size:", hvd.size())  # 输出 h + v + d 的尺寸
    output1 = output
    print("Output size:", output1.size())  # 输出 a 的尺寸

