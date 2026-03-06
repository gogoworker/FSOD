import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn


"""《Adaptive Rectangular Convolution for Remote Sensing Pansharpening》 CVPR 2025
基于卷积神经网络 (CNN) 的遥感全色锐化技术的最新进展显著提高了图像质量。然而，这些方法中的传统卷积模块有两个关键缺点。首先，卷积操作中的采样位置局限于固定的方形窗口。其次，采样点的数量是预设的，保持不变。
鉴于遥感图像中物体的大小各异，这些僵化的参数导致特征提取不理想。为了克服这些限制，我们引入了一个创新的卷积模块，自适应矩形卷积 (ARConv)。ARConv 自适应地学习卷积核的高度和宽度，并根据学习到的比例动态调整采样点的数量。
这种方法使 ARConv 能够有效地捕获图像中各种物体的特定尺度特征，从而优化内核大小和采样位置。此外，我们提出了 ARNet，这是一种以 ARConv 为主要卷积模块的网络架构。
对多个数据集的广泛评估表明，我们的方法在增强全色锐化性能方面优于以前的技术。消融研究和可视化进一步证实了ARConv的功效。
"""
# B站：箫张跋扈 整理并修改(https://space.bilibili.com/478113245)

class ARConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, flag=False, modulation=True):
        super(ARConv, self).__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
        )
        # 第一部分：卷积核高度和宽度的学习过程
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 输出高度特征图，范围在 (0, 1)
        )

        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 输出宽度特征图，范围在 (0, 1)
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.hook_handles = []
        self.hook_handles.append(self.m_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.m_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[1].register_full_backward_hook(self._set_lr))

        self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()  # 移除钩子函数
        self.hook_handles.clear()  # 清空句柄列表

    def forward(self, x, epoch, hw_range):
        assert isinstance(hw_range, list) and len(
            hw_range) == 2, "hw_range should be a list with 2 elements, represent the range of h w"
        scale = hw_range[1] // 9
        if hw_range[0] == 1 and hw_range[1] == 3:
            scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * (hw_range[1] - 1) + 1  # 高度特征图，范围在 [1, hw_range[1]]
        w = self.w_conv(offset) * (hw_range[1] - 1) + 1  # 宽度特征图，范围在 [1, hw_range[1]]
        # 第二部分：卷积核采样点数量的选择过程
        if epoch <= 100:
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)  # 计算高度的平均值
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)  # 计算宽度的平均值
            N_X = int(mean_l // scale)  # 根据高度平均值选择采样点数
            N_Y = int(mean_w // scale)  # 根据宽度平均值选择采样点数

            def phi(x):
                if x % 2 == 0:
                    x -= 1  # 确保采样点数为奇数
                return x

            N_X, N_Y = phi(N_X), phi(N_Y)  # 调整采样点数为奇数
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)  # 确保最小采样点数为 3
            N_X, N_Y = min(N_X, 7), min(N_Y, 7)  # 确保最大采样点数为 7
            if epoch == 100:
                self.reserved_NXY = nn.Parameter(
                    torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
                    requires_grad=False  # 固定采样点数
                )
        else:
            N_X = self.reserved_NXY[0]
            N_Y = self.reserved_NXY[1]

        # 第三部分：采样图的生成过程
        N = N_X * N_Y  # 总采样点数
        l = l.repeat([1, N, 1, 1])  # 重复高度特征图
        w = w.repeat([1, N, 1, 1])  # 重复宽度特征图
        offset = torch.cat((l, w), dim=1)  # 合并高度和宽度特征图
        dtype = offset.data.type()

        # 生成采样点位置
        p = self._get_p(offset, dtype, N_X, N_Y)  # (b, 2*N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)

        # 双线性插值
        q_lt = p.detach().floor()  # 左下角采样点
        q_rb = q_lt + 1  # 右上角采样点
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # 左上角采样点
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # 右下角采样点

        # 计算双线性插值权重
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )

        # 生成采样图
        x_q_lt = self._get_x_q(x, q_lt, N)  # 左下角采样值
        x_q_rb = self._get_x_q(x, q_rb, N)  # 右上角采样值
        x_q_lb = self._get_x_q(x, q_lb, N)  # 左上角采样值
        x_q_rt = self._get_x_q(x, q_rt, N)  # 右下角采样值

        x_offset = (
                g_lt.unsqueeze(dim=1) * x_q_lt
                + g_rb.unsqueeze(dim=1) * x_q_rb
                + g_lb.unsqueeze(dim=1) * x_q_lb
                + g_rt.unsqueeze(dim=1) * x_q_rt
        )  # 双线性插值结果
        # 第四部分：卷积操作的实现
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)  # 调整采样图的形状
        x_offset = self.dropout2(x_offset)  # 随机丢弃部分采样点
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)  # 使用对应的卷积核进行卷积
        out = x_offset * m + bias  # 输出最终的特征图
        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype, n_x, n_y):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
                .unsqueeze(dim=1)
                .expand(-1, c, -1, -1, -1)
                .contiguous()
                .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset

if __name__ == '__main__':
    block = ARConv(inc=3, outc=3).to('cuda')

    input_tensor = torch.rand(1, 3, 32, 32).to('cuda')

    # 设置 epoch 和 hw_range 作为测试参数
    epoch = 50  # 假设当前 epoch 为 50
    hw_range = [1, 3]  # 假设 hw_range 为 [1, 3] , 用来计算比例因子

    output = block(input_tensor, epoch, hw_range)

    print("输入大小:", input_tensor.size())
    print("输出大小:", output.size())
