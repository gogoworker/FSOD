import math
import torch
import torch.nn as nn
from typing import Literal
from einops import rearrange
from einops.layers.torch import Rearrange



class AttentionBlock(nn.Module):
    """
    Global multi-head self-attention block with optional projection.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = (
            dim_head * heads
        )  # the total dimension used inside the multi-head attention. When concatenating all heads, the combined dimension is dim_head × heads
        project_out = not (
            heads == 1 and dim_head == dim
        )  # if we're using just 1 head and its dimension equals dim, then we can skip the final linear projection.

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)  # Applies LN over the last dimension.

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Expected input shape: [B, L, C]
        """
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # chunk splits into 3 chuncks along the last dimension, this gives Q, K, V
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalAttention2D(nn.Module):
    """
    Windowed/local attention for 2D grids using unfold & fold.
    """

    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride  # kernel_size
        self.dim = dim
        padding = 0

        self.norm = nn.LayerNorm(dim)

        self.Attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = rearrange(
            x, "B H W C -> B C H W"
        )  # Rearrange to [B, C, H, W] for unfolding

        # unfold into local 2D patches
        patches = self.unfold(x)  # [B, C*K*K, L] where W is the number of patches

        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        patches = self.norm(patches)

        # Intra-Window self.attention
        out = self.Attention(patches)  # [B*L, K*K, C]

        # Reshape back to [B, C*K*K, L]
        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        # Fold back to [B, C, H, W] with overlap
        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)

        # Normalize overlapping regions
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))  # [B, K*K, L]
        norm = fold(norm)  # [B, 1, H, W]
        out = out / norm

        # Reshape to [B, H, W, C]
        out = rearrange(out, "B C H W -> B H W C")

        return out


class MultipoleAttention(nn.Module):
    """
    Hierarchical local attention across multiple scales with down/up-sampling.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        local_attention_kernel_size,
        local_attention_stride,
        downsampling: Literal["avg_pool", "conv"],
        upsampling: Literal["avg_pool", "conv"],
        sampling_rate,
        heads,
        dim_head,
        dropout,
        channel_scale,
    ):
        super().__init__()

        self.levels = int(math.log(image_size, sampling_rate))  # math.log(x, base)

        channels_conv = [in_channels * (channel_scale**i) for i in range(self.levels)]

        # A shared local attention layer for all levels
        self.Attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=channels_conv[0],
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )

        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "avg_pool":
            current = image_size

            for _ in range(self.levels):
                assert (
                    current % sampling_rate == 0
                ), f"Image size not divisible by sampling_rate size at level {_}: current={current}, sampling_ratel={sampling_rate}"
                current = current // sampling_rate

            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )

        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )

    def forward(self, x):
        # x: [B, H, W, C], returns the same shape
        # Level 0
        x_in = x

        x_out = []
        x_out.append(self.Attention(x_in))

        # Levels from 1 to L
        for l in range(1, self.levels):
            x_in = self.down(x_in)
            x_out_down = self.Attention(x_in)
            x_out.append(x_out_down)

        res = x_out.pop()
        for l, out_down in enumerate(x_out[::-1]):
            res = out_down + (1 / (l + 1)) * self.up(res)

        return res


if __name__ == '__main__':
    image_size = 16  # 输入图像大小（2^4=16，采样率2时共4个层级）
    in_channels = 4  # 输入通道数（与dim_head=4匹配）
    # 窗口尺寸设置为2：确保最小层级（16→8→4→2，共4层）的2x2特征图能容纳窗口
    local_attention_kernel_size = 2
    local_attention_stride = 2  # 步长与窗口一致，无重叠且不遗漏
    downsampling = "avg_pool"
    upsampling = "avg_pool"
    sampling_rate = 2  # 采样率（2的幂次，确保图像尺寸可整除）
    heads = 1
    dim_head = 4
    dropout = 0.1
    channel_scale = 1

    block = MultipoleAttention(
        image_size=image_size,
        in_channels=in_channels,
        local_attention_kernel_size=local_attention_kernel_size,
        local_attention_stride=local_attention_stride,
        downsampling=downsampling,
        upsampling=upsampling,
        sampling_rate=sampling_rate,
        heads=heads,
        dim_head=dim_head,
        dropout=dropout,
        channel_scale=channel_scale
    ).to('cuda')

    # 创建测试输入 [B, H, W, C]
    batch_size = 2
    input_tensor = torch.randn(batch_size, image_size, image_size, in_channels).to('cuda')

    output_tensor = block(input_tensor)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")
