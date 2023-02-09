import math
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from equalized import EqualLinear
from fused_act import FusedLeakyReLU
from upfirdn2d import Upfirdn2dUpsample, Upfirdn2dBlur

# mapping子网的第一层
class PixelNorm(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * paddle.rsqrt(
            paddle.mean(inputs * inputs, 1, keepdim=True) + 1e-8)

# 论文中实验的结果是：style会过度缩放，所谓的调制（Mod）就是改weights值，论文中用了标准差来合理化缩放
# modulation、conv、norm三个功能模块组合成的子网
class ModulatedConv2D(nn.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Upfirdn2dBlur(blur_kernel,
                                      pad=(pad0, pad1),
                                      upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * (kernel_size * kernel_size)
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = self.create_parameter(
            (1, out_channel, in_channel, kernel_size, kernel_size),
            default_initializer=nn.initializer.Normal())

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})")

    def forward(self, inputs, style):
        batch, in_channel, height, width = inputs.shape

        style = self.modulation(style).reshape((batch, 1, in_channel, 1, 1))
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = paddle.rsqrt((weight * weight).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.reshape((batch, self.out_channel, 1, 1, 1))

        weight = weight.reshape((batch * self.out_channel, in_channel,
                                 self.kernel_size, self.kernel_size))

        if self.upsample:
            inputs = inputs.reshape((1, batch * in_channel, height, width))
            weight = weight.reshape((batch, self.out_channel, in_channel,
                                     self.kernel_size, self.kernel_size))
            weight = weight.transpose((0, 2, 1, 3, 4)).reshape(
                (batch * in_channel, self.out_channel, self.kernel_size,
                 self.kernel_size))
            out = F.conv2d_transpose(inputs,
                                     weight,
                                     padding=0,
                                     stride=2,
                                     groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))
            out = self.blur(out)

        elif self.downsample:
            inputs = self.blur(inputs)
            _, _, height, width = inputs.shape
            inputs = inputs.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(inputs, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        else:
            inputs = inputs.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(inputs, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        return out

# 噪声B处理工具类
class NoiseInjection(nn.Layer):
    def __init__(self, is_concat=False):
        super().__init__()

        self.weight = self.create_parameter(
            (1, ), default_initializer=nn.initializer.Constant(0.0))
        self.is_concat = is_concat

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = paddle.randn((batch, 1, height, width))
        if self.is_concat:
            return paddle.concat([image, self.weight * noise], axis=1)
        else:
            return image + self.weight * noise

# 作者实验发现：CGAN系列算法的初始输入没有必要用高斯噪声了，因此改成了常量
class ConstantInput(nn.Layer):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = self.create_parameter(
            (1, channel, size, size),
            default_initializer=nn.initializer.Normal())

    def forward(self, inputs):
        batch = inputs.shape[0]
        out = self.input.tile((batch, 1, 1, 1))

        return out

# 论文中定义的Style Block的子模块，1个Block有2个StyleConv
class StyledConv(nn.Layer):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 style_dim,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 is_concat=False):
        super().__init__()

        self.conv = ModulatedConv2D(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection(is_concat=is_concat)
        self.activate = FusedLeakyReLU(out_channel *
                                       2 if is_concat else out_channel)

    def forward(self, inputs, style, noise=None):
        out = self.conv(inputs, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out

# ToRGB保证输出到下一层的通道数是3
# 在PGGAN中使用的是1×1的conv，这里做了一个小改进，用ModulatedConv2D代替，效果类似
class ToRGB(nn.Layer):
    def __init__(self,
                 in_channel,
                 style_dim,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upfirdn2dUpsample(blur_kernel)

        self.conv = ModulatedConv2D(in_channel,
                                    3,
                                    1,
                                    style_dim,
                                    demodulate=False)
        self.bias = self.create_parameter((1, 3, 1, 1),
                                          nn.initializer.Constant(0.0))

    def forward(self, inputs, style, skip=None):
        out = self.conv(inputs, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out