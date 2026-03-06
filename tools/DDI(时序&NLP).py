import torch
import torch.nn as nn
import torch.nn.functional as F

import math

"""《Adaptive multi-scale decomposition framework for time series forecasting》 AAAI 2026
基于 Transformer 和 MLP 的方法已成为时间序列预测 (TSF) 领域的主流方法。然而，现实世界的时间序列通常在不同尺度上呈现出不同的模式，而未来的变化则由这些重叠尺度的相互作用所塑造，因此需要高容量的模型。
虽然基于 Transformer 的方法擅长捕捉长程依赖关系，但它们的计算复杂度较高，并且容易过拟合。相反，基于 MLP 的方法在时间动态建模方面提供了计算效率和熟练度，但它们难以有效地捕捉复杂尺度的时间模式。
基于对时间序列中多尺度纠缠效应的观察，我们提出了一种基于 MLP 的自适应多尺度分解 (AMD) 框架，用于 TSF。我们的框架将时间序列分解为多个尺度上不同的时间模式，并利用多尺度可分解混合 (MDM) 模块来剖析和聚合这些模式。
我们的方法结合了双重依赖交互 (Dual Dependency Interaction，DDI) 模块和自适应多预测器合成 (AMS) 模块，能够有效地建模时间和通道依赖性，并利用自相关性来优化多尺度数据集成。
全面的实验表明，我们的 AMD 框架不仅克服了现有方法的局限性，而且在各种数据集上始终保持最佳性能。
"""


class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=2, alpha=0.0, layernorm=True):
        super(DDI, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.input_shape = input_shape
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.n_history = 1
        self.alpha = alpha
        self.patch = patch

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        # [batch_size, feature_num, seq_len]
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # input [batch_size, feature_num, self.n_history * patch]
            input = output[:, :, i - self.n_history * self.patch: i]
            # input [batch_size, feature_num, self.n_history * patch]
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)
            # aggregation
            # [batch_size, feature_num, patch]
            input = F.gelu(self.agg(input))  # self.n_history * patch -> patch
            input = self.dropout_t(input)
            # input [batch_size, feature_num, patch]
            # input = torch.squeeze(input, dim=-1)
            tmp = input + x[:, :, i: i + self.patch]

            res = tmp

            # [batch_size, feature_num, patch]
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                # [batch_size, patch, feature_num]
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)
            output[:, :, i: i + self.patch] = res + self.alpha * tmp

        # [batch_size, feature_num, seq_len]
        return output

if __name__ == '__main__':
    batch_size = 32
    seq_len = 128
    feature_num = 64

    # [batch_size, feature_num, seq_len]
    input = torch.rand(batch_size, feature_num, seq_len).to('cuda')

    block = DDI(input_shape=(seq_len, feature_num)).to('cuda')

    output = block(input)

    print(f"Input size: {input.size()}")
    print(f"Output size: {output.size()}")
