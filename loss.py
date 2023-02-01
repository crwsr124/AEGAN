from models.paddle2torch import *
import math

smooth_l1_loss = nn.SmoothL1Loss()
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
bce_logits_loss = nn.BCEWithLogitsLoss()

def Standardization(x):
    # mean = x.mean(1, keepdim=True)
    # return x - mean
    mean = x.mean(1, keepdim=True)
    std = x.std(1, keepdim=True)
    return (x - mean)/(std+1e-8)

# def RpGANLoss_d(fake_logit, real_logit):
#     ones = torch.ones_like(real_logit)
#     loss = bce_logits_loss(real_logit - fake_logit, ones)
#     return loss
def RpGANLoss_d(fake_logit, real_logit):
    return (F.relu(1 + (fake_logit - real_logit))).mean() #+ real_logit.abs().mean()

def FocalRpGANLoss_d(fake_logit, real_logit):
    kkkk = 1.-F.relu((F.sigmoid((real_logit - fake_logit).detach())-0.5))*2.
    ones = torch.ones_like(real_logit)
    loss = kkkk**4 * torch.nn.BCEWithLogitsLoss(reduction='none')(real_logit - fake_logit, ones)
    return loss.mean()

# def RpGANLoss_g(fake_logit, real_logit):
#     ones = torch.ones_like(real_logit)
#     loss = bce_logits_loss(fake_logit - real_logit, ones)
#     return loss
def RpGANLoss_g(fake_logit, real_logit):
    return (F.relu(1 + (real_logit - fake_logit))).mean()

def RaGANLoss_d(fake_logit, real_logit):
    ones = torch.ones_like(real_logit)
    zeros = torch.zeros_like(real_logit)
    fakeline = real_logit - torch.mean(real_logit, 0, keepdim=True) + torch.mean(fake_logit, 0, keepdim=True)
    realline = fake_logit - torch.mean(fake_logit, 0, keepdim=True) + torch.mean(real_logit, 0, keepdim=True)
    loss = bce_logits_loss(real_logit - fakeline.detach(), ones) + bce_logits_loss(fake_logit - realline.detach(), zeros)
    kkkk = torch.mean(real_logit) - torch.mean(fake_logit)
    kkkk = 1.-F.relu((F.sigmoid(kkkk)-0.5))*2.
    return kkkk**4*(0.5*loss + 0.05*(((real_logit - torch.mean(real_logit, 0, keepdim=True))**2).mean() + ((fake_logit - torch.mean(fake_logit, 0, keepdim=True))**2).mean()))

def RaGANLoss_g(fake_logit, real_logit):
    ones = torch.ones_like(real_logit)
    zeros = torch.zeros_like(real_logit)
    #fakeline = (real_logit - torch.mean(real_logit, dim=0, keepdim=True)).detach() + torch.mean(fake_logit, dim=0, keepdim=True)
    realline = fake_logit - torch.mean(fake_logit, 0, keepdim=True) + torch.mean(real_logit, 0, keepdim=True)
    loss = bce_logits_loss(fake_logit - realline.detach(), ones) #+ bce_stable(real_logit - fakeline, zeros)
    return loss #+ torch.mean(fake_logit).abs()