import torch
import torch.nn as nn

"""《FilterNet: Harnessing Frequency Filters for Time Series Forecasting》NIPS2024
虽然已经提出了许多使用不同的网络架构的预测器，但基于 Transformer 的模型在时间序列预测方面具有最先进的性能。
然而，基于 Transformer 的预报器仍然受到高频信号脆弱性、计算效率和全谱利用率瓶颈的困扰，这本质上是准确预测数千个点的时间序列的基石。
在本文中，我们探讨了一种用于深度时间序列预测的启发性信号处理的新视角。
受过滤过程的启发，我们引入了一个简单而有效的网络，即 FilterNet，它建立在我们提出的可学习频率滤波器之上，通过选择性地传递或衰减时间序列信号的某些组成部分来提取关键的信息时间模式。
具体来说，我们在 FilterNet 中提出了两种可学习的滤波器：（i） 平面整形滤波器，采用通用频率内核进行信号过滤和时间建模;（ii） 上下文整形滤波器，利用过滤频率来检查其与输入信号的兼容性进行依赖学习。
配备这两个滤波器，FilterNet 可以近似替代时间序列文献中广泛采用的线性映射和注意力映射，同时在处理高频噪声和利用有利于预测的整个频谱方面具有卓越的能力。
最后，我们对 8 个时间序列预测基准进行了广泛的实验，实验结果表明，与最先进的方法相比，我们在有效性和效率方面的表现都非常出色。代码可从以下存储库获得：https://github.com/aikunyi/FilterNet
"""

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class PaiFilter(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, hidden_size):
        super(PaiFilter, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = hidden_size

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)

        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x


if __name__ == '__main__':
    seq_len = 12
    pred_len = 12  # 预测长度
    enc_in = 16
    hidden_size = 64  # 隐藏层大小

    model = PaiFilter(seq_len, pred_len, enc_in, hidden_size)

    # (batch_size, seq_len, enc_in)
    input_tensor = torch.rand(32, seq_len, enc_in)

    x_mark_enc = torch.zeros_like(input_tensor)  # 假设编码的时间标记
    x_mark_dec = torch.zeros_like(input_tensor)  # 假设解码的时间标记
    output = model(input_tensor, x_mark_enc, input_tensor, x_mark_dec)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
