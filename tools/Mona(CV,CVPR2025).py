import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

"""《5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks》 CVPR 2025
预训练和微调可以提升视觉任务的迁移效率和性能。近期的增量调优方法为视觉分类任务提供了更多选择。尽管现有的视觉增量调优方法取得了成功，但在目标检测和分割等挑战性任务上，它们仍未能突破完全微调的上限。
为了找到一种能够取代完全微调的竞争性方法，我们提出了一种基于适配器的新型调优方法——多认知视觉适配器 (Multi-cognitive Visual Adapter, Mona)。
首先，我们在适配器中引入多个视觉友好的滤波器，以增强其处理视觉信号的能力，而之前的方法主要依赖于语言友好的线性滤波器。其次，我们在适配器中添加了缩放归一化层，以调节视觉滤波器输入特征的分布。
为了充分展示 Mona 的实用性和通用性，我们在多个代表性视觉任务上进行了实验，包括 COCO 数据集上的实例分割、ADE20K 数据集上的语义分割、Pascal VOC 数据集上的目标检测、DOTA/STAR 数据集上的定向目标检测以及三个常见数据集上的图像分类。
令人兴奋的结果表明，Mona 在所有这些任务上都超越了完全微调，并且是唯一一个在上述各种任务上表现优于完全微调的增量微调方法。例如，与完全微调相比，Mona 在 COCO 数据集上实现了 1% 的性能提升。
综合结果表明，Mona 微调比完全微调更适合保留和利用预训练模型的功能。
"""


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

class Mona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2


if __name__ == '__main__':
    block = Mona(in_dim=256).to('cuda')

    # 假设 batch_size=2, sequence_length=16 (h=4, w=4), feature_dim=256
    batch_size = 2
    h, w = 4, 4
    seq_len = h * w
    feature_dim = 256
    input_tensor = torch.rand(batch_size, seq_len, feature_dim).to('cuda')

    output = block(input_tensor, hw_shapes=(h, w))

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())


