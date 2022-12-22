from paddle2torch import *


def DiffMask(x, max_ratio=0.01):
    width = x.shape[2]//16
    ratio = max_ratio*torch.rand([1])
    # ratio = max_ratio
    mask = ((torch.rand([x.shape[0], 1, width, width])-ratio).ceil()).repeat_interleave(x.shape[1],1)
    mask = nn.Upsample(scale_factor=(16,16), mode='bicubic')(mask)
    x = x * mask
    return x


if __name__ == "__main__":
    img = torch.randn((2, 3, 256, 256))
    kkk = DiffMask(img)
    print(kkk.shape)