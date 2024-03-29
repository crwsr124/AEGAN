

import math
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

# 引入通用工具类
from equalized import EqualLinear
from fused_act import FusedLeakyReLU
from upfirdn2d import Upfirdn2dUpsample, Upfirdn2dBlur

from g_utils import PixelNorm, ConstantInput, StyledConv, ToRGB


class Generator(nn.Layer):
    def __init__(self,
                 size,
                 style_dim,
                 n_mlp,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 is_concat=False):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(style_dim,
                            style_dim,
                            lr_mul=lr_mlp,
                            activation="fused_lrelu"))

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4],
                                self.channels[4],
                                3,
                                style_dim,
                                blur_kernel=blur_kernel,
                                is_concat=is_concat)
        self.to_rgb1 = ToRGB(self.channels[4] *
                             2 if is_concat else self.channels[4],
                             style_dim,
                             upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.LayerList()
        self.upsamples = nn.LayerList()
        self.to_rgbs = nn.LayerList()
        self.noises = nn.Layer()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f"noise_{layer_idx}",
                                        paddle.randn(shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel * 2 if is_concat else in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    is_concat=is_concat,
                ))

            self.convs.append(
                StyledConv(out_channel * 2 if is_concat else out_channel,
                           out_channel,
                           3,
                           style_dim,
                           blur_kernel=blur_kernel,
                           is_concat=is_concat))

            self.to_rgbs.append(
                ToRGB(out_channel * 2 if is_concat else out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        self.is_concat = is_concat

    def make_noise(self):
        noises = [paddle.randn((1, 1, 2**2, 2**2))]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(paddle.randn((1, 1, 2**i, 2**i)))

        return noises

    def mean_latent(self, n_latent):
        latent_in = paddle.randn((n_latent, self.style_dim))
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, inputs):
        return self.style(inputs)

    def get_mean_style(self):
        mean_style = None
        with paddle.no_grad():
            for i in range(10):
                style = self.mean_latent(1024)
                if mean_style is None:
                    mean_style = style
                else:
                    mean_style += style

        mean_style /= 10
        return mean_style

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1.0,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            # styles = [self.style(s) for s in styles]
            styles = [self.style(styles)]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}")
                    for i in range(self.num_layers)
                ]

        if truncation < 1.0:
            style_t = []
            if truncation_latent is None:
                truncation_latent = self.get_mean_style()
            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).tile((1, inject_index, 1))

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).tile((1, inject_index, 1))
            latent2 = styles[1].unsqueeze(1).tile(
                (1, self.n_latent - inject_index, 1))

            latent = paddle.concat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        if self.is_concat:
            noise_i = 1

            outs = []
            for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2],
                                            self.to_rgbs):
                out = conv1(out, latent[:, i],
                            noise=noise[(noise_i + 1) // 2])  ### 1 for 2
                out = conv2(out,
                            latent[:, i + 1],
                            noise=noise[(noise_i + 2) // 2])  ### 1 for 2
                skip = to_rgb(out, latent[:, i + 2], skip)

                i += 2
                noise_i += 2
        else:
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2],
                    self.to_rgbs):
                out = conv1(out, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)

                i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            # return image, None
            return image

if __name__ == "__main__":
    G = Generator(256, 512, 8)
    # print(G)
    gauss_noise = paddle.randn([2, 512])
    out = G(gauss_noise)
    print(out.shape)
