import torch
import torch.nn as nn


from einops.einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out

class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

class FFM(nn.Module):
    def __init__(self, dim1, dim2, res):
        super().__init__()
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        self.concat = nn.Conv2d(dim2*2, dim2, 1)

        self.fusion = nn.Sequential(IDSC(dim2*4, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    DSC(dim2, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    nn.Conv2d(dim2, dim2, 1),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU())


    def forward(self, x, y):
        b, c, h, w = x.shape
        B, N, C = y.shape
        H = W = int(N**0.5)

        x = self.trans_c(x)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)

        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        qy = self.qy(y).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        kx = self.kx(x).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        vx = self.vx(x).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C**-0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H//4, w//4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B,  H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)


        qx = self.qx(x).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        ky = self.ky(y).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        vy = self.vy(y).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)

        attny = (qx @ ky.transpose(-2, -1)) * (C**-0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H//4, w//4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)

        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)

        out = torch.cat([x, y, out1, out2], dim=1)

        out = self.fusion(out)
        return out


if __name__ == '__main__':

    batch_size = 2
    c_in = 64  #
    c_out = 64
    height = 32
    width = 32
    N = height * width

    x_input = torch.rand(batch_size, c_in, height, width).to('cuda')  # For x input
    y_input = torch.rand(batch_size, N, c_in).to('cuda')  # For y input

    block = FFM(dim1=c_in, dim2=c_out, res=height).to('cuda')

    output = block(x_input, y_input)

    print("x_input size:", x_input.size())
    print("y_input size:", y_input.size())
    print("Output size:", output.size())
