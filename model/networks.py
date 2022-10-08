from msilib import sequence
import torch.nn.functional as F
import torch.nn as nn
import functools

class Generator(nn.Module):
    def __init__(self, in_c, out_c, nf, params):
        super().__init__()

        self.params = params
        self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        self.in_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, nf, kernel_size=7, bias=False),
            self.norm_layer(nf),
        )

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            self.norm_layer(nf * 2),
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            self.norm_layer(nf * 4),
        )

        res_blocks = []
        for i in range(params.r_blocks):
            res_blocks += [ResidualBlock(nf, self.norm_layer)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            self.norm_layer(nf * 2),
        )

        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            self.norm_layer(nf),
        )

        self.out_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, out_c, kernel_size=7)
        )
    
    def forward(self, x):
        x = F.relu(self.in_block(x))
        x = F.relu(self.downsample_1(x))
        x = F.relu(self.downsample_2(x))
        x = self.res_blocks(x)
        x = F.relu(self.upsample_1(x))
        x = F.relu(self.upsample_2(x))
        out = F.tanh(self.out_block(x))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, nf, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, bias=False),
            norm_layer(nf * 4)
        )
    
    def forward(self, x):
        r = F.relu(self.block(x))
        out = self.block(r) + x
        return out

class NLayerDiscriminator(nn.Module):
    def __init__(self, in_c, nf, params):
        super().__init__()

        k = 4
        s = 2
        p = 1
        sequence = []
        self.params = params
        self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        sequence += [nn.Conv2d(in_c, nf, kernel_size=k, stride=s, padding=p, bias=False)]

        mul = 1
        mul_prev = 1
        for n in range(1, params.n_layers+1):
            if n == params.n_layers:
                s = 1
            mul_prev = mul
            mul = min(2 ** n, 8)
            sequence += [nn.Conv2d(nf * mul_prev, nf * mul, kernel_size=k, stride=s, padding=p, bias=False),
                        self.norm_layer(nf * mul),
                        nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(nf * mul, 1, kernel_size=k, stride=s, padding=p, bias=False)]

        self.sequence = nn.Sequential(*sequence)

    def forward(self, x):
        return self.sequence(x)
