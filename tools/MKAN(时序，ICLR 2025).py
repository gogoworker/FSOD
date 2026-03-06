import torch
import torch.nn as nn


"""《TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting》 ICLR 2025
现实世界的时间序列通常具有多个相互交织的频率成分，这使得准确的时间序列预测具有挑战性。将混合频率成分分解为多个单频率成分是一种自然的选择。
然而，模式的信息密度在不同频率上有所不同，对不同频率成分采用统一的建模方法可能会导致不准确的表征。
为了应对这一挑战，受最近的 Kolmogorov-Arnold 网络 (KAN) 灵活性的启发，我们提出了一种基于 KAN 的频率分解学习架构 (TimeKAN)，以解决由多频率混合引起的复杂预测挑战。
具体来说，TimeKAN 主要由三个组件组成：级联频率分解 (CFD) 块、多阶 KAN 表示学习 (M-KAN) 块和频率混合块。CFD 块采用自下而上的级联方法获得每个频带的序列表示。
得益于 KAN 的高灵活性，我们设计了一个新颖的 M-KAN 块来学习和表示每个频带内的特定时间模式。最后，使用频率混合块将频带重新组合为原始格式。
在多个真实世界时间序列数据集上进行的大量实验结果表明，TimeKAN 作为一种极轻量级架构实现了最先进的性能。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # View and repeat input degree + 1 times
        b,c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:,::2]
            mul_2 = x[:,1::2]
            mul_res = mul_1 * mul_2
            x = torch.concat([x[:,:x.shape[1]//2], mul_res])
        x = x.view((b, c_in, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.acos(x)
        # x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        # # Multiply by arange [0 .. degree]
        x = x* self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.post_mul:
            mul_1 = y[:,::2]
            mul_2 = y[:,1::2]
            mul_res = mul_1 * mul_2
            y = torch.concat([y[:,:y.shape[1]//2], mul_res])
        return y



class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
            in_features,
            out_features,
            order)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        return x



class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree, stride=1, padding=0, dilation=1, groups=1, act=False, bn=False,
                 bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class M_KAN(nn.Module):
    def __init__(self, d_model, seq_len, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            ChebyKANLayer(d_model, d_model, order)
        )
        self.conv = BasicConv(d_model, d_model, kernel_size=3, degree=order, groups=d_model)

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out = x1 + x2
        return out


if __name__ == '__main__':
    batch_size = 16
    seq_len = 32
    d_model = 64
    order = 3

    block = M_KAN(d_model=d_model, seq_len=seq_len, order=order).to('cuda')

    input = torch.rand(batch_size, seq_len, d_model).to('cuda')

    output = block(input)

    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")


