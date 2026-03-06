import torch
import torch.nn as nn

"""《Frequency Adaptive Normalization For Non-stationary Time Series Forecasting》arxiv2024
时间序列预测通常需要处理具有不断变化的趋势和季节性模式的非平稳数据。为了解决非平稳性问题，最近提出了可逆实例规一化，以通过某些统计指标（例如均值和方差）减轻趋势的影响。
尽管它们表现出更高的预测准确性，但它们仅限于表达基本趋势，无法处理季节性模式。为了解决这一限制，本文提出了一种新的实例规一化解决方案，称为频率自适应规一化 (FAN)，它扩展了实例规范化以处理动态趋势和季节性模式。
具体来说，我们使用傅里叶变换来识别涵盖大多数非平稳因素的实例主要频繁成分。此外，输入和输出之间这些频率成分的差异被明确建模为具有简单 MLP 模型的预测任务。
FAN 是一种与模型无关的方法，可以应用于任意预测主干。我们在四种广泛使用的预测模型上实例化 FAN 作为主干，并在八个基准数据集上评估它们的预测性能改进。 
FAN 表现出显著的性能提升，MSE 平均提升了 7.76% ~ 37.90%。
"""

def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered


class MLPfreq(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)


class FAN(nn.Module):
    """FAN first substract bottom k frequecy component from the original series


    Args:
        nn (_type_): _description_
    """

    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True, **kwargs): # 指定从输入实例的频率域中选取的前 K 个主导频率分量的数量
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        print("freq_topk : ", self.freq_topk)
        self.rfft = rfft

        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def loss(self, true):
        # freq normalization
        residual, pred_main = main_freq_part(true, self.freq_topk, self.rfft)

        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)

    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape
        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)

        return norm_input.reshape(bs, len, dim)

    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(bs, len, dim)

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)

if __name__ == '__main__':
    seq_len = 64
    pred_len = 32
    enc_in = 8

    block = FAN(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in).to('cuda')

    input_tensor = torch.rand(2, seq_len, enc_in).to('cuda')

    # 正向传播，'n'表示进行归一化
    output_tensor = block(input_tensor, mode='n')

    print("Input tensor size:", input_tensor.size())
    print("Output tensor size:", output_tensor.size())