import torch
import torch.nn as nn


class PIM(nn.Module):
    def __init__(self, channel):
        super(PIM, self).__init__()

        self.processmag = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0))

        self.processpha = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0))

    def forward(self, rgb_x, ycbcr_x):
        rgb_fft = torch.fft.rfft2(rgb_x, norm='backward')
        ycbcr_fft = torch.fft.rfft2(ycbcr_x, norm='backward')
        rgb_amp = torch.abs(rgb_fft)
        rgb_phase = torch.angle(rgb_fft)

        ycbcr_amp = torch.abs(ycbcr_fft)
        ycbcr_phase = torch.angle(ycbcr_fft)

        rgb_amp = self.processmag(rgb_amp)
        rgb_phase = self.processmag(rgb_phase)

        ycbcr_amp = self.processmag(ycbcr_amp)
        ycbcr_phase = self.processmag(ycbcr_phase)

        mix_phase = rgb_phase + ycbcr_phase

        out_rgb = torch.fft.irfft2(rgb_amp * torch.exp(1j * mix_phase), norm='backward')
        out_ycbcr = torch.fft.irfft2(ycbcr_amp * torch.exp(1j * mix_phase), norm='backward')
        return out_rgb, out_ycbcr


if __name__ == '__main__':
    channel = 3
    block = PIM(channel).to('cuda')

    input_rgb = torch.rand(1, channel, 64, 64).to('cuda')
    input_ycbcr = torch.rand(1, channel, 64, 64).to('cuda')

    output_rgb, output_ycbcr = block(input_rgb, input_ycbcr)

    print(f'Input RGB size: {input_rgb.size()}')
    print(f'Input YCbCr size: {input_ycbcr.size()}')
    print(f'Output RGB size: {output_rgb.size()}')
    print(f'Output YCbCr size: {output_ycbcr.size()}')