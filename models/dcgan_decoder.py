# import torch
# import torch.nn as nn

from paddle2torch import *

class SeWeight(nn.Module):
    def __init__(self, in_size, out_size, reduction=4):
        super(SeWeight, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_size // reduction),
            nn.LayerNorm([in_size // reduction, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(in_size // reduction, out_size, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_size),
            nn.LayerNorm([out_size, 1, 1]),
            # nn.Hardsigmoid()
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x)

class SEConvTranspose2d(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding, bias=True):
        super(SEConvTranspose2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.ConvTranspose2d(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding),dim=None)
        self.sew = SeWeight(inc, outc, reduction=4)

    def forward(self, x):
        return self.conv(x)*self.sew(x)

class DCGenerator(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(DCGenerator, self).__init__()

        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            # Deconv layer
            if i == 0:
                deconv = SEConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0, bias=False)
            else:
                deconv = SEConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1, bias=False)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            # BN layer
            bn_name = 'bn' + str(i + 1)
            self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, nn.ReLU())

        self.output_layer = nn.Sequential(
            SEConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # style = x.view((x.shape[0], -1, 1, 1))
        style = x.reshape((x.shape[0], -1, 1, 1))
        h = self.hidden_layer(style)
        out = self.output_layer(h)
        return out

def toNumpyWithNorm(img_tensor):
    out = torch.reshape(img_tensor, (img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]))
    out = out.detach().numpy().transpose((1, 2, 0))
    out = out*0.5+0.5
    return out

# from thop import profile
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    net = DCGenerator(512, [512, 256, 128, 64, 64, 64], 3)
    net.eval()
    
    latent = torch.randn((2, 512))
    for i in range(10):
        out = net(latent)
        print(out.shape)

    # million = 100 * 10000
    # FLOPs = torch.flops(net, [1, 256], print_detail=True)
    # FLOPs, _ = profile(net, (style,))
    # print("Decoder FLOPs:", FLOPs/million)

    # img_out = toNumpyWithNorm(out)

    # plt.figure("haha")
    # plt.subplot(1,2,1), plt.title('1')
    # plt.imshow(img_out)
    # plt.subplot(1,2,2), plt.title('2')
    # plt.imshow(img_out2)
    # plt.show()