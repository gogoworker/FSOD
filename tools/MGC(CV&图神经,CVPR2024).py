import torch
from torch import nn
from timm.models.layers import DropPath
import warnings
warnings.filterwarnings('ignore')


"""《Scaling Graph Convolutions for Mobile Vision》 CVPR 2024
为了与现有的移动架构竞争，MobileViG 引入了稀疏视觉图注意力 (SVGA)，这是一种基于 GNN 原理的快速令牌混合运算符。然而，MobileViG 的扩展性很差，模型大小最多比具有相似延迟的模型落后 1%。
本文介绍了移动图卷积 (MGC)，这是一种解决此扩展问题的新视觉图神经网络 (ViG) 模块。我们提出的移动视觉架构 MobileViGv2 使用 MGC 来证明我们方法的有效性。MGC 通过增加图稀疏性并在图操作中引入条件位置编码来改进 SVGA。
我们最小的模型 MobileViGv2-Ti 在 ImageNet-1K 上实现了 77.7% 的 top-1 准确率，比 MobileViG-Ti 高 2%，在 iPhone 13 Mini NPU 上的推理延迟为 0.9 毫秒。
我们最大的模型 MobileViGv2-B 实现了 83.4% 的 top-1 准确率，比 MobileViG-B 高 0.8%，推理延迟为 2.7 毫秒。除了图像分类之外，我们还表明 MobileViGv2 在其他任务上也能很好地推广。
对于 MS COCO 2017 上的对象检测和实例分割，MobileViGv2-M 的表现优于 MobileViG-M 1.2 AP^ box 和 0.7 AP^ mask，而 MobileViGv2-B 的表现优于 MobileViG-B 1.0 AP^ box 和 0.7 AP^ mask。
对于 ADE20K 上的语义分割，MobileViGv2-M 实现了 42.9% mIoU，而 MobileViGv2-B 实现了 44.3% mIoU。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)

class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape

        '''
        This is the original SVGA graph construction
        '''
        # x_j = x - x
        # x_m = x
        # cnt = 1
        # for i in range(self.K, H, self.K):
        #     x_c = torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
        #     # x_m += x_c
        #     # cnt += 1
        #     x_j = torch.max(x_j, x_c - x)
        # for i in range(self.K, W, self.K):
        #     x_r = torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
        #     # x_m += x_r
        #     # cnt += 1
        #     x_j = torch.max(x_j, x_r - x)

        '''
        This is the 5 connection graph construction
        '''
        x_j = x - x
        x_c = torch.cat([x[:, :, -self.K:, :], x[:, :, :-self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c - x)
        x_c = torch.cat([x[:, :, self.K:, :], x[:, :, :self.K, :]], dim=2)
        x_j = torch.max(x_j, x_c - x)
        x_r = torch.cat([x[:, :, :, -self.K:], x[:, :, :, :-self.K]], dim=3)
        x_j = torch.max(x_j, x_r - x)
        x_r = torch.cat([x[:, :, :, self.K:], x[:, :, :, :self.K]], dim=3)
        x_j = torch.max(x_j, x_r - x)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class RepCPE(nn.Module):
    """
    This implementation of reparameterized conditional positional encoding was originally implemented
    in the following repository: https://github.com/apple/ml-fastvit

    Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
            self,
            in_channels,
            embed_dim,
            spatial_shape=(7, 7),
            inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor):
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self):
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, K):
        super(Grapher, self).__init__()
        self.cpe = RepCPE(in_channels=in_channels, embed_dim=in_channels, spatial_shape=(7, 7))
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels * 2, in_channels, K=K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)

        return x


class MGC(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.mixer = Grapher(in_dim, K)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # 初始化DropPath模块，当drop_path>0时使用随机路径丢弃，否则使用恒等映射
        self.use_layer_scale = use_layer_scale # 设置是否使用层缩放标志
        if use_layer_scale: # 如果启用层缩放
            self.layer_scale_1 = nn.Parameter( # 定义第一个可学习的层缩放参数
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True)  # 初始化为指定值的(in_dim,1,1)张量，设为可训练
            self.layer_scale_2 = nn.Parameter( # 定义第二个可学习的层缩放参数
                layer_scale_init_value * torch.ones((in_dim, 1, 1)), requires_grad=True) # 同上，用于另一个分支的缩放


    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


if __name__ == '__main__':
    # 创建MGC模块实例，假设输入通道数为64
    in_dim = 64
    block = MGC(in_dim=in_dim).to('cuda')

    # 创建符合预期的4D输入张量 (batch_size, channels, height, width)
    input = torch.rand(1, in_dim, 32, 32).to('cuda')  # 假设输入是32x32的特征图

    # 前向传播
    output = block(input)

    # 打印输入输出形状
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)
