import torch
import math
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


class DecayPos1d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen):
        '''
        slen: (c)
        recurrent is not implemented
        '''
        mask_c = self.generate_1d_decay(slen)
        retention_rel_pos = mask_c

        return retention_rel_pos


class VolSelfAttention(nn.Module):
    r""" Volumetric Self-Attention"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        ##
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # spectrum location prior
        # 64 128 256 512
        self.realPos = DecayPos1d(64, num_heads, 2, 4)

        ##
        self.qkv_C = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv_C = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        # self.conv_point_C = nn.Conv2d(dim*3, dim*3, kernel_size=1, stride=1, padding=0, groups=1)
        self.proj_C = nn.Conv2d(dim, dim, kernel_size=1)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.Gao_spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_heads, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1),
        )

        self.Gao_channel_attention = nn.Sequential(
            # 48 30 6 30
            nn.Conv2d(dim // num_heads, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        ##Spatial-wise Projection.
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)  ##W-MSA
        if mask is not None:  ##W-MSA
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  ##W-MSA

        x1 = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        # spectrum location prior
        realPos = self.realPos(c / self.num_heads)

        ## Spectrum-wise Projection.
        x_s = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        qkv_c = self.qkv_dwconv_C(self.qkv_C(x_s))
        q_c, k_c, v_c = qkv_c.chunk(3, dim=1)

        q_c = rearrange(q_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_c = rearrange(k_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_c = rearrange(v_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_c = torch.nn.functional.normalize(q_c, dim=-1)
        k_c = torch.nn.functional.normalize(k_c, dim=-1)

        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature + realPos
        attn_c = attn_c.softmax(dim=-1)

        x2 = (attn_c @ v_c)

        x2 = rearrange(x2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=hh, w=hh)

        x2 = self.proj_C(x2)
        x2 = rearrange(x2, ' b c h w -> b (h w) c', h=hh, w=hh)

        # VolAtt
        attn_spatial = attn
        attn_spatial = self.Gao_spatial_attention(attn_spatial)
        attn_b, _, _, _ = attn_spatial.shape
        attn_spatial = attn_spatial.reshape(attn_b, hh * hh, 1)
        x4 = attn_spatial * x2

        x5 = x1 + x2 + x4
        return x5

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


if __name__ == '__main__':
    # 设置参数
    dim = 64
    window_size = (8, 8)
    num_heads = 4  # 注意力头数

    block = VolSelfAttention(dim=dim, window_size=window_size, num_heads=num_heads).to('cuda')

    batch_size = 2
    num_patches_per_window = window_size[0] * window_size[1]
    input_tensor = torch.randn(batch_size, num_patches_per_window, dim).to('cuda')

    output = block(input_tensor).to('cuda')

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")