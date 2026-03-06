import torch
import torch.nn as nn
import torch.nn.functional as F

"""《FAN: Fourier Analysis Networks》
尽管神经网络，特别是以 MLP 和 Transformer 为代表的神经网络取得了显着的成功，但我们揭示了它们在周期性的建模和推理方面表现出潜在的缺陷，
即它们倾向于记住周期性数据，而不是真正理解周期性的基本原理。
然而，周期性是各种形式的推理和泛化的关键特征，通过观察中的重复模式支撑自然系统和工程系统的可预测性。
在本文中，我们提出了 FAN，这是一种基于傅里叶分析的新型网络架构，它能够有效地对周期性现象进行建模和推理。
通过引入傅里叶级数，周期性自然地集成到神经网络的结构和计算过程中，从而实现对周期性模式的更准确表达和预测。
作为多层感知器 （MLP） 的有前途的替代品，FAN 可以在各种模型中无缝替代 MLP，参数和 FLOP 更少。
通过广泛的实验，我们证明了 FAN 在周期函数建模和推理方面的有效性，以及 FAN 在一系列实际任务中的优越性和泛化性，包括符号公式表示、时间序列预测和语言建模。
"""

class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True). 
            There is almost no difference between bias and non-bias in our experiments.
    """

    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()

        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms

        # Linear transformation for the p component (for cosine and sine parts)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)

        # Linear transformation for the g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """

        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))

        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)

        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)

        return output

if __name__ == '__main__':
    input = torch.rand(8,16)
    block = FANLayer(input_dim=16, output_dim=16, p_ratio=0.25, activation='gelu', use_p_bias=True)
    output = block(input)
    print(input.size())
    print(output.size())