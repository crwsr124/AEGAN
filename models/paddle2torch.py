import paddle
import paddle as torch
import paddle.nn as nn
import paddle.nn.functional as F


nn.ConvTranspose2d = nn.Conv2DTranspose
nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2D
nn.Conv2d = nn.Conv2D
nn.Conv1d = nn.Conv1D
nn.BatchNorm2d = nn.BatchNorm2D
nn.InstanceNorm2d = nn.InstanceNorm2D
nn.UpsamplingBilinear2d = nn.UpsamplingBilinear2D

nn.init = nn.initializer
nn.init.kaiming_normal_ = nn.initializer.KaimingNormal
nn.init.constant_ = nn.initializer.Constant
nn.init.orthogonal_ = nn.initializer.Orthogonal

nn.Module = nn.Layer
nn.Module.add_module = nn.Layer.add_sublayer
nn.Module.modules = nn.Layer.sublayers

torch.Tensor.view = paddle.Tensor.reshape
nn.Parameter = paddle.create_parameter

torch.optim = paddle.optimizer
torch.optim.AdamW.zero_grad = paddle.optimizer.AdamW.clear_grad