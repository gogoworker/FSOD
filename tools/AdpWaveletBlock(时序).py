import torch
import torch.nn as nn
import torch.nn.functional as F

"""《AdaWaveNet: Adaptive Wavelet Network for Time Series Analysis》Arxiv2024
时间序列数据分析是金融、医疗保健和气象等各个领域的关键组成部分。尽管深度学习在时间序列分析方面取得了进展，但在解决时间序列数据的非平稳性方面仍然存在挑战。
传统模型建立在统计特性随时间恒定的假设之上，通常难以捕捉现实时间序列中的时间动态，导致时间序列分析出现偏差和错误。
本文介绍了自适应小波网络 (AdaWaveNet)，这是一种采用自适应小波变换对非平稳时间序列数据进行多尺度分析的新方法。
AdaWaveNet 设计了一种基于提升方案的小波分解和构造机制，用于自适应和可学习的小波变换，从而提高了分析的灵活性和鲁棒性。
我们对 10 个数据集进行了广泛的实验，涉及 3 个不同的任务，包括预测、归纳和新建立的超分辨率任务。
评估证明了 AdaWaveNet 在这三个任务中比现有方法更有效，这说明了它在各种实际应用中的潜力。
"""

class Splitting(nn.Module):
    def __init__(self, channel_first):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.channel_first = channel_first
        if (channel_first):
            self.conv_even = lambda x: x[:, :, ::2]
            self.conv_odd = lambda x: x[:, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, ::2, :]
            self.conv_odd = lambda x: x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))


class LiftingScheme(nn.Module):
    def __init__(self, in_channels, input_size, modified=True, splitting=True, k_size=4, simple_lifting=True):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        kernel_size = k_size
        pad = (k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(channel_first=True)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.GELU(),
                nn.LayerNorm([in_channels, input_size // 2])
            ]
            modules_U += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=in_channels),
                nn.GELU(),
                nn.LayerNorm([in_channels, input_size // 2])
            ]
        else:
            size_hidden = 2

            modules_P += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels * prev_size, in_channels * size_hidden, kernel_size=kernel_size, stride=1,
                          groups=in_channels),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_channels * prev_size, in_channels * size_hidden, kernel_size=kernel_size, stride=1,
                          groups=in_channels),
                nn.Tanh()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv1d(in_channels * prev_size, in_channels, kernel_size=1, stride=1, groups=in_channels),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv1d(in_channels * prev_size, in_channels, kernel_size=1, stride=1, groups=in_channels),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d)
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)


def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


class AdpWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(AdpWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size, input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        (c, d) = self.wavelet(x)

        # Upsample c and d to match the input size
        c = F.interpolate(c, size=x.size(2), mode='linear', align_corners=True)
        d = F.interpolate(d, size=x.size(2), mode='linear', align_corners=True)

        r = None
        if (self.regu_approx + self.regu_details != 0.0):
            if self.regu_details:
                rd = self.regu_details * d.abs().mean()
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc

        x = self.norm_x(c)  # Use the upsampled c as the output
        d = self.norm_d(d)

        return x, r, d # x：信号的低频部分，表示经过小波变换后的近似部分。r：正则化项，衡量信号中低频部分和高频部分之间的差异。 d：信号的高频部分，表示细节部分或噪声。


class Config:
    def __init__(self):
        self.enc_in = 64  # 输入通道数
        self.lifting_kernel_size = 4  # 卷积核大小
        self.regu_details = 0.1  # 细节部分的正则化强度
        self.regu_approx = 0.1  # 近似部分的正则化强度


if __name__ == '__main__':
    configs = Config()

    # (batch_size, channels, sequence_length)
    batch_size = 2
    input_size = 128  # 序列长度
    input_tensor = torch.rand(batch_size, configs.enc_in, input_size)

    block = AdpWaveletBlock(configs, input_size)

    x, r, d = block(input_tensor)

    print(f"Input size: {input_tensor.size()}")
    print(f"Output x size: {x.size()}")
    print(f"Output r size: {r if r is not None else 'None'}")
    print(f"Output d size: {d.size()}")