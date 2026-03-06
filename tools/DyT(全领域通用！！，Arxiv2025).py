import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


"""《TransformerswithoutNormalization》 Arxiv2025
归一化层在现代神经网络中无处不在，长期以来一直被认为是必不可少的。这项工作表明，无需归一化的 Transformer 模型可以通过一种非常简单的技术实现相同甚至更好的性能。
我们引入了动态 Tanh（Dynamic Tanh，DyT），这是一种逐元素操作 DyT(x)=tanh(αx)，作为 Transformer 中归一化层的即插即用替代方案。
DyT 的灵感来源于对 Transformer 中层归一化的观察，即层归一化通常会产生类似 tanh 的 S 形输入-输出映射。
通过引入 DyT，无需归一化的 Transformer 模型可以匹配甚至超越其归一化对应模型的性能，且大多数情况下无需超参数调优。我们在从识别到生成、从监督学习到自监督学习、
从计算机视觉到语言模型等多种场景中验证了带有 DyT 的 Transformer 的有效性。这些发现挑战了传统观念，即归一化层在现代神经网络中是必不可少的，并为它们在深度网络中的作用提供了新的见解。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module  # 初始化输出模块为输入模块
    if isinstance(module, nn.LayerNorm):  # 检查当前模块是否为 LayerNorm 类型
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))  # 如果是 LayerNorm，替换为 DynamicTanh
    for name, child in module.named_children():  # 遍历当前模块的所有子模块
        module_output.add_module(name, convert_ln_to_dyt(child))  # 递归调用，替换子模块中的 LayerNorm
    del module  # 删除原模块，释放内存
    return module_output  # 返回替换后的模块


if __name__ == '__main__':
    normalized_shape = 128
    channels_last = True

    block = DynamicTanh(normalized_shape, channels_last).to('cuda')


    input = torch.rand(32, 784, 128).to('cuda')

    # 前向传播
    output = block(input)

    # 打印输入和输出的形状
    print("Input size:", input.size())
    print("Output size:", output.size())