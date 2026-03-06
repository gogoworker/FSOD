import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange



class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.pdim = pdim  # 表示输入数据的前半部分通道数。在 forward 中，输入张量会被拆分成两部分，pdim 决定了拆分的比例。
        self.lk_size = kernel_size  # 用于定义静态 LK（长程）卷积核的大小
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(   # 动态卷积核生成的子模块
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=bs * self.pdim)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)

            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=self.lk_size // 2) + x1_

            x = torch.cat([x1, x2], dim=1)
        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(self.pdim, 1, self.sk_size, self.sk_size)
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=self.lk_size // 2) \
                               + F.conv2d(x[:, :self.pdim], dynamic_kernel, stride=1, padding=self.sk_size // 2,
                                          groups=self.pdim)
        return x

    def extra_repr(self):
        return f'pdim={self.pdim}'


if __name__ == '__main__':
    batch_size = 1
    channels = 8
    height = 32
    width = 32
    pdim = 4

    input_tensor = torch.rand(batch_size, channels, height, width).to('cuda')

    # Create a random lk_filter tensor with size (pdim, pdim, kernel_size, kernel_size)
    lk_filter = torch.rand(pdim, pdim, 13, 13).to('cuda')  # Changed kernel size to match class default (13)

    block = ConvolutionalAttention(pdim=pdim).to('cuda')

    output = block(input_tensor, lk_filter)

    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')