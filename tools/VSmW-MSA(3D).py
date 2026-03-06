import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import DropPath, trunc_normal_


"""《
VSmTrans: A hybrid paradigm integrating self-attention and convolution
for 3D medical image segmentation
》Medical Image Analysis   1区TOP
目的： 近年来，视觉变换器（Vision Transformers，ViTs）由于其在学习全局表示方面的出色能力，已经在与卷积神经网络（CNNs）相比中取得了竞争力的性能。
然而，将它们应用于3D图像分割时面临两个主要挑战：i）由于3D医学图像的巨大尺寸，由于巨大的计算成本，全面的全局信息很难捕获；
ii）变换器在局部归纳偏差方面的不足影响了其分割细节特征的能力，例如模糊的边界和细微的定义。
因此，为了将视觉变换器机制应用于医学图像分割领域，需要充分克服上述挑战。

方法： 我们提出了一种混合范式，称为可变形状混合变换器（VSmTrans），它将自注意力和卷积结合在一起，
能够享受自注意力机制中复杂关系的自由学习以及卷积中的局部先验知识的好处。
具体来说，我们设计了一种可变形状的自注意力机制(variable-shape mixed window multi-head self-attention,可变形状混合窗口多头自注意力)，能够在不增加额外计算成本的情况下快速扩展感受野，并实现全局意识与局部细节之间的良好平衡。
此外，平行卷积范式引入了强大的局部归纳偏差，促进了细节的挖掘能力。同时，一对可学习的参数可以自动调整上述两种范式的重要性。
我们在两个不同模态的公共医学图像数据集上进行了广泛的实验：AMOS CT 数据集和 BraTS2021 MRI 数据集。

结果： 我们的方法在这些数据集上获得了最佳的平均 Dice 分数，分别为 88.3% 和 89.7%，优于之前基于 Swin Transformer 和 CNN 的最先进架构。
还进行了系列消融实验，以验证提出的混合机制及其组件的效率，并探索 VSmTrans 中这些关键参数的有效性。

结论： 所提出的基于混合变换器的3D医学图像分割骨干网络能够紧密结合自注意力和卷积，充分利用这两种范式的优势。
实验结果证明了我们方法的优越性，相较于其他最先进的方法，混合范式似乎最适合医学图像分割领域。
消融实验也表明，提出的混合机制能够有效平衡大感受野与局部归纳偏差，从而带来高度准确的分割结果，特别是在捕捉细节方面。
我们的代码可以在 https://github.com/qingze-bai/VSmTrans 获取。
"""


def window_partition(x, D_sp, H_sp, W_sp, num_heads=None, is_Mask=False):
    B, D, H, W, C = x.shape
    if is_Mask:
        x = x.reshape(B, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp, C).contiguous()
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, D_sp * H_sp * W_sp, C)
    else:
        x = x.reshape(B, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp, C // num_heads, num_heads).contiguous()
        x = x.permute(0, 1, 3, 5, 8, 2, 4, 6, 7).contiguous().view(-1, num_heads, D_sp * H_sp * W_sp, C // num_heads)
    return x

def window_reverse(x, D_sp, H_sp, W_sp, D, H, W):
    _, _, C = x.shape
    x = x.view(-1, D // D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, C).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(-1, D, H, W, C).contiguous()
    return x

def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size[0], window_size[1], window_size[2], is_Mask=True)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class VariableShapeAttention(nn.Module):
    def __init__(self, feature_size, idx, split_size, window_size, num_head, img_size, shift=False, attn_drop_rate=0.):
        super(VariableShapeAttention, self).__init__()
        self.num_head = num_head
        self.init_window_size(idx, img_size, split_size, window_size)
        head_dim = 4 * feature_size // num_head
        self.scale = head_dim ** -0.5
        self.shift = shift
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

        mesh_args = torch.meshgrid.__kwdefaults__
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.D_sp - 1) * (2 * self.H_sp - 1) * (2 * self.W_sp - 1),
                num_head,
            )
        )
        coords_d = torch.arange(self.D_sp)
        coords_h = torch.arange(self.H_sp)
        coords_w = torch.arange(self.W_sp)
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.D_sp - 1
        relative_coords[:, :, 1] += self.H_sp - 1
        relative_coords[:, :, 2] += self.W_sp - 1
        relative_coords[:, :, 0] *= (2 * self.H_sp - 1) * (2 * self.W_sp - 1)
        relative_coords[:, :, 1] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_window_size(self, idx, img_size, split_size, window_size):
        if idx == 0:
            self.D_sp, self.H_sp, self.W_sp = (
                window_size[0] if img_size[0] > window_size[0] else img_size[0],
                window_size[1] if img_size[1] > window_size[1] else img_size[1],
                window_size[2] if img_size[2] > window_size[2] else img_size[2],
            )
            self.D_sf, self.H_sf, self.W_sf = (
                self.D_sp // 2 if img_size[0] > self.D_sp else 0,
                self.H_sp // 2 if img_size[1] > self.H_sp else 0,
                self.W_sp // 2 if img_size[2] > self.W_sp else 0,
            )
        elif idx == 1:
            self.D_sp, self.H_sp, self.W_sp = (
                split_size[0] if img_size[0] > split_size[0] else img_size[0],
                img_size[1],
                split_size[2] if img_size[2] > split_size[2] else img_size[2],
            )
            self.D_sf, self.H_sf, self.W_sf = (
                self.D_sp // 2 if img_size[0] > self.D_sp else 0,
                0,
                self.W_sp // 2 if img_size[2] > self.W_sp else 0,
            )
        elif idx == 2:
            self.D_sp, self.H_sp, self.W_sp = (
                split_size[0] if img_size[0] > split_size[0] else img_size[0],
                split_size[1] if img_size[1] > split_size[1] else img_size[1],
                img_size[2],
            )
            self.D_sf, self.H_sf, self.W_sf = (
                self.D_sp // 2 if img_size[0] > self.D_sp else 0,
                self.H_sp // 2 if img_size[1] > self.H_sp else 0,
                0,
            )
        elif idx == 3:
            self.D_sp, self.H_sp, self.W_sp = img_size[0], \
                                              split_size[1] if img_size[1] > split_size[1] else img_size[1], \
                                              split_size[2] if img_size[2] > split_size[2] else img_size[2]
            self.D_sf, self.H_sf, self.W_sf = 0, \
                                              self.H_sp // 2 if img_size[1] > self.H_sp else 0, \
                                              self.W_sp // 2 if img_size[2] > self.W_sp else 0

    def forward(self, qkv):
        B, D, H, W, C = qkv.shape
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.D_sp - D % self.D_sp) % self.D_sp
        pad_b = (self.H_sp - H % self.H_sp) % self.H_sp
        pad_r = (self.W_sp - W % self.W_sp) % self.W_sp
        qkv = F.pad(qkv, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = qkv.shape

        if self.shift:
            qkv = torch.roll(qkv, shifts=(-self.D_sf, -self.H_sf, -self.W_sf), dims=(1, 2, 3))

        qkv = qkv.reshape(B, Dp, Hp, Wp, 3, C // 3).permute(4, 0, 1, 2, 3, 5)
        q = window_partition(qkv[0], self.D_sp, self.H_sp, self.W_sp, self.num_head)
        k = window_partition(qkv[1], self.D_sp, self.H_sp, self.W_sp, self.num_head)
        v = window_partition(qkv[2], self.D_sp, self.H_sp, self.W_sp, self.num_head)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        n = self.D_sp * self.H_sp * self.W_sp
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if self.shift:
            mask = compute_mask(dims=[Dp, Hp, Wp], window_size=(self.D_sp, self.H_sp, self.W_sp),
                                shift_size=(self.D_sf, self.H_sf, self.W_sf), device=qkv.device)
            nw = mask.shape[0]
            attn = attn.view(attn.shape[0] // nw, nw, self.num_head, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_head, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.D_sp * self.H_sp * self.W_sp, C // 3).contiguous()
        x = window_reverse(x, self.D_sp, self.H_sp, self.W_sp, Dp, Hp, Wp)
        if self.shift:
            x = torch.roll(x, shifts=(self.D_sf, self.H_sf, self.W_sf), dims=(1, 2, 3))

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

class VSmixWindow_MSA(nn.Module):
    def __init__(self,
                 feature_size,
                 split_size,
                 window_size,
                 num_head,
                 img_size,
                 shift=False,
                 qkv_bias=False,
                 attn_drop_rate=0.0,
                 drop_rate=0.0):
        super(VSmixWindow_MSA, self).__init__()
        self.num_head = num_head
        self.qkv = nn.Linear(feature_size, feature_size * 3, bias=qkv_bias)
        self.act1 = nn.GELU()
        self.conv1 = nn.Linear(feature_size * 3, feature_size)
        self.norm1 = nn.LayerNorm(feature_size, eps=1e-6)
        self.dep_conv = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(num_features=feature_size)
        self.act2 = nn.LeakyReLU()

        self.attns = nn.ModuleList([
            VariableShapeAttention(
                feature_size=feature_size // 4,
                idx=i % 4,
                split_size=split_size,
                window_size=window_size,
                num_head=num_head,
                img_size=img_size,
                shift=shift,
                attn_drop_rate=attn_drop_rate
            )
            for i in range(4)])

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.drop = nn.Dropout(drop_rate)
        self.reset_parameters()

        self.proj = nn.Linear(feature_size, feature_size)
        self.proj_drop = nn.Dropout(drop_rate)

    def reset_parameters(self):
        if self.rate1 is not None:
            self.rate1.data.fill_(0.5)
        if self.rate2 is not None:
            self.rate2.data.fill_(0.5)
        self.dep_conv.bias.data.fill_(0.0)

    def forward(self, x):
        qkv = self.qkv(x)
        B, D, H, W, C = qkv.shape
        # Conv
        # B, D, H, W, C
        conv_x = self.conv1(self.act1(qkv))
        conv_x = self.norm1(conv_x).permute(0, 4, 1, 2, 3)
        conv_x = self.dep_conv(conv_x)
        conv_x = self.act2(self.norm2(conv_x)).permute(0, 2, 3, 4, 1)
        # Transformer
        x1 = self.attns[0](qkv[:, :, :, :, :C // 4])
        x2 = self.attns[1](qkv[:, :, :, :, C // 4:C // 4 * 2])
        x3 = self.attns[2](qkv[:, :, :, :, C // 4 * 2:C // 4 * 3])
        x4 = self.attns[3](qkv[:, :, :, :, C // 4 * 3:])
        attn_x = torch.cat([x1, x2, x3, x4], dim=-1)
        # 考虑注意力通道问题
        attn_x = self.proj_drop(self.proj(attn_x))
        x = self.rate1*attn_x+self.rate2*conv_x
        x = self.drop(x)
        return x

if __name__ == '__main__':
    batch_size = 2
    D = 8  # Depth (D)
    H = 8  # Height (H)
    W = 8  # Width (W)
    feature_size = 16
    split_size = (8, 8, 8)  # 切分大小，确保是一个三维元组
    window_size = (8, 8, 8)  # 窗口尺寸
    num_head = 4
    img_size = (D, H, W)  # 图像尺寸
    shift = False  # 是否启用移动

    block = VSmixWindow_MSA(
        feature_size=feature_size,
        split_size=split_size,  # 传递三维元组
        window_size=window_size,
        num_head=num_head,
        img_size=img_size,
        shift=shift,
        qkv_bias=True,
        attn_drop_rate=0.1,
        drop_rate=0.1
    )

    input_tensor = torch.rand(batch_size, D, H, W, feature_size)

    output = block(input_tensor)

    print(f"Input size: {input_tensor.size()}")
    print(f"Output size: {output.size()}")
