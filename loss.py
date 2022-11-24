import torch
import math

device = 'cuda'

smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
l1_loss = torch.nn.L1Loss().to(device)
mse_loss = torch.nn.MSELoss().to(device)
bce_logits_loss = torch.nn.BCEWithLogitsLoss().to(device)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    def forward(self, input, target):
        pt = torch.sigmoid(input)
        logp = self.ce(input, target)
        loss = (1 - pt) ** self.gamma * logp
        return loss.mean()

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        loss = -(1 - pt) ** self.gamma * target * torch.log(pt) - pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        loss = torch.mean(loss)
        return loss


def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).contiguous().view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

#      sum is not good for d_r1_loss

def d_r1_loss_mean(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.mean(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).contiguous().view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def standard_normalize(x):
    x_mu = torch.mean(x, dim=1, keepdim=True)
    x_std = torch.std(x, dim=1, keepdim=True)
    return (x-x_mu)/(x_std+1e-12)

# def pearson_correlation(x, y):
#     x = x - torch.mean(x, dim=1, keepdims=True)
#     y = y - torch.mean(y, dim=1, keepdims=True)
#     x = torch.nn.functional.normalize(x, p=2, dim=1)
#     y = torch.nn.functional.normalize(y, p=2, dim=1)
#     return torch.sum(x * y, dim=1, keepdims=True)

def pearson_correlation(x, y):
    x_mu = torch.mean(x, dim=1, keepdim=True)
    y_mu = torch.mean(y, dim=1, keepdim=True)
    x_std = torch.std(x, dim=1, keepdim=True)
    y_std = torch.std(y, dim=1, keepdim=True)
    a = torch.mean((x - x_mu) * (y - y_mu), dim=1, keepdim=True)
    b = x_std * y_std
    return a / b

def loss_function_original(recon_x, x, mu, logvar):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


class DWT(torch.nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH

class DWTLoss(torch.nn.Module):
    def __init__(self):
        super(DWTLoss, self).__init__()
        self.dwt = DWT()
        self.l1 = torch.nn.L1Loss()

    def forward(self, x, x_g):
        x_LL1, x_HL1, x_LH1, x_HH1 = self.dwt(x)
        x_LL2, x_HL2, x_LH2, x_HH2 = self.dwt(x_LL1)
        x_LL3, x_HL3, x_LH3, x_HH3 = self.dwt(x_LL2)

        x_LL1_g, x_HL1_g, x_LH1_g, x_HH1_g = self.dwt(x_g)
        x_LL2_g, x_HL2_g, x_LH2_g, x_HH2_g = self.dwt(x_LL1_g)
        x_LL3_g, x_HL3_g, x_LH3_g, x_HH3_g = self.dwt(x_LL2_g)

        loss = self.l1(x_HL1, x_HL1_g) + self.l1(x_LH1, x_LH1_g) + self.l1(x_HH1, x_HH1_g) +\
            self.l1(x_HL2, x_HL2_g) + self.l1(x_LH2, x_LH2_g) + self.l1(x_HH2, x_HH2_g) +\
            self.l1(x_HL3, x_HL3_g) + self.l1(x_LH3, x_LH3_g) + self.l1(x_HH3, x_HH3_g)
        return loss

DWT_loss = DWTLoss()