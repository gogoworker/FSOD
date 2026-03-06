import warnings
warnings.filterwarnings("ignore")
import torch.nn.init as init
import numbers
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter


"""《Probing Synergistic High-Order Interaction in Infrared and Visible ImageFusion》 CVPR 2024
红外与可见光图像融合旨在通过整合和区分来自多个源的互补信息来生成融合图像。虽然具有全局空间交互的交叉注意力机制看似前景光明，但它仅能捕捉二阶空间交互，而忽略了空间和通道维度上的高阶交互。
这一局限性阻碍了多模态图像协同效应的发挥。为了弥合这一差距，我们引入了一种协同高阶交互范式 (Synergistic High-order Interaction Paradigm，SHIP)，旨在系统地研究红外与可见光图像在两个基本维度上的空间细粒度和全局统计协作：
1) 空间维度：我们通过数学上等同于全局交互的元素乘法构建空间细粒度交互，然后通过迭代聚合和演化互补信息来构建高阶格式，从而提高效率和灵活性。
2) 通道维度：在具有一阶统计量（平均值）的通道交互基础上，我们设计了高阶通道交互，以便于基于全局统计量识别源图像之间的相互依赖关系。
利用高阶交互显著增强了我们的模型利用多模式协同作用的能力，从而通过在各种基准上的综合实验证明了其优于最先进的替代方案的性能。
"""

class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class cdcconv(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.8):

        super(cdcconv, self).__init__()

        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.d_conv = Conv2d_Diag_Cross(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)

        self.HP_branch = Parameter(torch.FloatTensor(1))

    def forward(self, x):
        out1 = self.h_conv(x)
        out2 = self.d_conv(x)
        out = torch.sigmoid(self.HP_branch) * out1 + (1 - torch.sigmoid(self.HP_branch)) * out2 + x
        # out = out1 + out2
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class Refine(nn.Module):

    def __init__(self,n_feat,out_channels):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
             CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class DenseBlockMscale(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier'):
        super(DenseBlockMscale, self).__init__()
        self.ops = DenseBlock(channel_in, channel_out, init)
        self.fusepool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel_out, channel_out, 1, 1, 0),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.fc1 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fuse = nn.Conv2d(3 * channel_out, channel_out, 1, 1, 0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
        x1 = self.ops(x1)
        x2 = self.ops(x2)
        x3 = self.ops(x3)
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        xattw = self.fusepool(x1 + x2 + x3)
        xattw1 = self.fc1(xattw)
        xattw2 = self.fc2(xattw)
        xattw3 = self.fc3(xattw)
        # x = x1*xattw1+x2*xattw2+x3*xattw3
        x = self.fuse(torch.cat([x1 * xattw1, x2 * xattw2, x3 * xattw3], 1))

        return x


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlockMscale(channel_in, channel_out, init)
            else:
                return DenseBlockMscale(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class spatialInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(spatialInteraction, self).__init__()
        self.reflashFused1 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashFused2 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashFused3 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashInfrared1 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashInfrared2 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashInfrared3 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )

        self.norm1 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm3 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm4 = LayerNorm(channelout, LayerNorm_type='WithBias')

    def forward(self, vis, inf, i, j):
        _, C, H, W = vis.size()

        vis_fft = torch.fft.rfft2(vis.float())
        inf_fft = torch.fft.rfft2(inf.float())

        atten = vis_fft * inf_fft
        atten = torch.fft.irfft2(atten, s=(H, W))
        atten = self.norm1(atten)
        fused_OneOrderSpa = atten * inf

        fused_OneOrderSpa = self.reflashFused1(fused_OneOrderSpa)
        fused_OneOrderSpa = self.norm2(fused_OneOrderSpa)
        infraredReflash1 = self.reflashInfrared1(inf)
        fused_twoOrderSpa = fused_OneOrderSpa * infraredReflash1

        fused_twoOrderSpa = self.reflashFused2(fused_twoOrderSpa)
        fused_twoOrderSpa = self.norm3(fused_twoOrderSpa)
        infraredReflash2 = self.reflashInfrared2(infraredReflash1)
        fused_threeOrderSpa = fused_twoOrderSpa * infraredReflash2

        fused_threeOrderSpa = self.reflashFused3(fused_threeOrderSpa)
        fused_threeOrderSpa = self.norm4(fused_threeOrderSpa)
        infraredReflash3 = self.reflashInfrared3(infraredReflash2)
        fused_fourOrderSpa = fused_threeOrderSpa * infraredReflash3

        fused = fused_fourOrderSpa + vis

        return fused, infraredReflash3


class channelInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(channelInteraction, self).__init__()
        self.chaAtten = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))
        self.reflashChaAtten1 = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                              nn.ReLU(),
                                              nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))
        self.reflashChaAtten2 = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                              nn.ReLU(),
                                              nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))
        self.reflashChaAtten3 = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                              nn.ReLU(),
                                              nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))

        self.reflashFused1 = nn.Sequential(
            nn.Conv2d(channelin * 2, channelout * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout * 2, channelout * 2, 3, 1, 1)
        )
        self.reflashFused2 = nn.Sequential(
            nn.Conv2d(channelin * 2, channelout * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout * 2, channelout * 2, 3, 1, 1)
        )
        self.reflashFused3 = nn.Sequential(
            nn.Conv2d(channelin * 2, channelout * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout * 2, channelout * 2, 3, 1, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.postprocess = nn.Sequential(InvBlock(DenseBlock, 2 * channelin, channelout),
                                         nn.Conv2d(2 * channelout, channelout, 1, 1, 0))

    def forward(self, vis, inf, i, j):
        vis_cat = torch.cat([vis, inf], 1)

        chanAtten = self.chaAtten(self.avgpool(vis_cat)).softmax(1)
        channel_response = self.chaAtten(self.avgpool(vis_cat))
        fused_OneOrderCha = vis_cat * chanAtten

        fused_OneOrderCha = self.reflashFused1(fused_OneOrderCha)
        chanAttenReflash1 = self.reflashChaAtten1(chanAtten).softmax(1)
        fused_twoOrderCha = fused_OneOrderCha * chanAttenReflash1

        fused_twoOrderCha = self.reflashFused2(fused_twoOrderCha)
        chanAttenReflash2 = self.reflashChaAtten2(chanAttenReflash1).softmax(1)
        fused_threeOrderCha = fused_twoOrderCha * chanAttenReflash2

        fused_threeOrderCha = self.reflashFused3(fused_threeOrderCha)
        chanAttenReflash3 = self.reflashChaAtten3(chanAttenReflash2).softmax(1)
        fused_fourOrderCha = fused_threeOrderCha * chanAttenReflash3

        fused_fourOrderCha = self.postprocess(fused_fourOrderCha)

        fused = fused_fourOrderCha + vis

        return fused, inf


class highOrderInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(highOrderInteraction, self).__init__()
        self.spatial = spatialInteraction(channelin, channelout)
        self.channel = channelInteraction(channelin, channelout)

    def forward(self, vis_y, inf, i, j):
        vis_spa, inf_spa = self.spatial(vis_y, inf, i, j)
        vis_cha, inf_cha = self.channel(vis_spa, inf_spa, i, j)

        return vis_cha, inf_cha


class EdgeBlock(nn.Module):
    def __init__(self, channelin, channelout):
        super(EdgeBlock, self).__init__()
        self.process = nn.Conv2d(channelin, channelout, 3, 1, 1)
        self.Res = nn.Sequential(nn.Conv2d(channelout, channelout, 3, 1, 1),
                                 nn.ReLU(), nn.Conv2d(channelout, channelout, 3, 1, 1))
        self.CDC = cdcconv(channelin, channelout)

    def forward(self, x):
        x = self.process(x)
        out = self.Res(x) + self.CDC(x)

        return out


class FeatureExtract(nn.Module):
    def __init__(self, channelin, channelout):
        super(FeatureExtract, self).__init__()
        self.conv = nn.Conv2d(channelin, channelout, 1, 1, 0)
        self.block1 = EdgeBlock(channelout, channelout)
        self.block2 = EdgeBlock(channelout, channelout)

    def forward(self, x):
        xf = self.conv(x)
        xf1 = self.block1(xf)
        xf2 = self.block2(xf1)

        return xf2


class SHIP(nn.Module):
    def __init__(self, vis_channels=1, inf_channels=1,
                 n_feat=16):
        super(SHIP, self).__init__()

        self.vis = FeatureExtract(vis_channels, n_feat)
        self.inf = FeatureExtract(inf_channels, n_feat)

        self.interaction1 = highOrderInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction2 = highOrderInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction3 = highOrderInteraction(channelin=n_feat, channelout=n_feat)

        self.postprocess = nn.Sequential(InvBlock(DenseBlock, 3 * n_feat, 3 * n_feat // 2),
                                         nn.Conv2d(3 * n_feat, n_feat, 1, 1, 0))

        self.reconstruction = Refine(n_feat, out_channels=vis_channels)

        self.i = 0

    def forward(self, image_vis, image_ir):
        vis_y = image_vis[:, :1]
        inf = image_ir

        vis_y = self.vis(vis_y)
        inf = self.inf(inf)

        vis_y_feat, inf_feat = self.interaction1(vis_y, inf, self.i, j=1)
        vis_y_feat2, inf_feat2 = self.interaction2(vis_y_feat, inf_feat, self.i, j=2)
        vis_y_feat3, inf_feat3 = self.interaction3(vis_y_feat2, inf_feat2, self.i, j=3)

        fused = self.postprocess(torch.cat([vis_y_feat, vis_y_feat2, vis_y_feat3], 1))

        fused = self.reconstruction(fused)

        self.i += 1

        return fused


if __name__ == '__main__':

    model = SHIP(vis_channels=1, inf_channels=1, n_feat=16).to('cuda')

    batch_size = 4
    img_height, img_width = 16, 16

    # 可见光图像（假设单通道）
    image_vis = torch.randn(batch_size, 1, img_height, img_width).to('cuda')
    # 红外图像（假设单通道）
    image_ir = torch.randn(batch_size, 1, img_height, img_width).to('cuda')


    output = model(image_vis, image_ir)

    # 打印输入输出尺寸
    print("可见光输入尺寸:", image_vis.shape)
    print("红外输入尺寸:", image_ir.shape)
    print("输出尺寸:", output.shape)