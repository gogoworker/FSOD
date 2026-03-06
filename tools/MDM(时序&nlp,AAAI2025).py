import torch
import torch.nn as nn

"""《Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting》AAAI 2025
基于 Transformer 和 MLP 的方法已成为时间序列预测 (TSF) 领域的主流方法。然而，现实世界的时间序列通常在不同尺度上呈现出不同的模式，而未来的变化则由这些重叠尺度的相互作用所塑造，因此需要高容量的模型。
虽然基于 Transformer 的方法擅长捕捉长程依赖关系，但它们的计算复杂度较高，并且容易过拟合。相反，基于 MLP 的方法在时间动态建模方面提供了计算效率和熟练度，但它们难以有效地捕捉复杂尺度的时间模式。
基于对时间序列中多尺度纠缠效应的观察，我们提出了一种基于 MLP 的自适应多尺度分解 (AMD) 框架，用于 TSF。我们的框架将时间序列分解为多个尺度上不同的时间模式，并利用多尺度可分解混合 (MDM) 模块来剖析和聚合这些模式。
我们的方法结合了双重依赖交互 (DDI) 模块和自适应多预测器合成 (AMS) 模块，能够有效地建模时间和通道依赖性，并利用自相关性来优化多尺度数据集成。
全面的实验表明，我们的 AMD 框架不仅克服了现有方法的局限性，而且在各种数据集上始终保持最佳性能。
"""

class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        self.seq_len = input_shape[0]
        self.k = k
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        # 如果启用了LayerNorm，首先对输入数据进行展平（flatten），然后进行BatchNorm
        if self.layernorm:  # 如果启用LayerNorm
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)  # 对输入展平后进行批归一化，再reshape回原始形状

        # 如果k为0，说明没有进行多尺度处理，直接返回输入数据
        if self.k == 0:  # 如果没有进行多尺度处理
            return x  # 直接返回原始输入

        # 创建一个空列表sample_x，用于存储不同尺度下池化后的数据
        sample_x = []  # 存储不同尺度池化后的数据

        # 对每个尺度的k，使用对应的池化层进行池化
        for i, k in enumerate(self.k_list):  # 遍历不同的尺度
            sample_x.append(self.avg_pools[i](x))  # 对每个尺度应用池化操作，将结果加入到sample_x列表中

        sample_x.append(x)  # 将原始输入x也加入sample_x，以便进行后续混合

        # 获取sample_x列表的长度，n表示池化后的数据数量
        n = len(sample_x)  # 获取sample_x的长度，即不同尺度的数据数量

        # 对每个尺度的特征进行处理，将不同尺度的结果加到下一尺度上
        for i in range(n - 1):  # 对池化后的数据进行混合
            tmp = self.linears[i](sample_x[i])  # 对每个尺度的数据进行线性变换

            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)  # 将变换后的数据与下一尺度的数据进行残差加法

        # 最终返回混合后的结果，即最后一个尺度的数据
        return sample_x[n - 1]  # 返回最后一个尺度的结果，经过多尺度混合后的最终特征


if __name__ == '__main__':
    batch_size = 32
    seq_len = 96
    feature_num = 8  # Number of features/channels

    mdm = MDM(input_shape=(seq_len, feature_num), k=3, c=2).to('cuda')

    # [batch_size, feature_num, seq_len]
    input_tensor = torch.randn(batch_size, feature_num, seq_len).to('cuda')

    output = mdm(input_tensor)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())