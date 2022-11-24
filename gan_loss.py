import torch
import math

device = 'cuda'

bce_stable = torch.nn.BCEWithLogitsLoss().to(device)

def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).contiguous().view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

# def RaGANLoss_d(fake_logit, real_logit):
#     ones = torch.ones_like(real_logit).to(device)
#     zeros = torch.zeros_like(real_logit).to(device)
#     loss = bce_stable(real_logit - torch.mean(fake_logit,dim=0,keepdim=True), ones) + bce_stable(fake_logit - torch.mean(real_logit,dim=0,keepdim=True), zeros)
#     return loss + torch.mean(real_logit).abs() + torch.mean(fake_logit).abs()


# def RaGANLoss_g(fake_logit, real_logit):
#     ones = torch.ones_like(real_logit).to(device)
#     zeros = torch.zeros_like(real_logit).to(device)
#     loss = bce_stable(real_logit - torch.mean(fake_logit,dim=0,keepdim=True), zeros) + bce_stable(fake_logit - torch.mean(real_logit,dim=0,keepdim=True), ones)
#     # return loss + torch.mean(fake_logit).abs()
#     return loss 

def RaGANLoss_d(fake_logit, real_logit):
    ones = torch.ones_like(real_logit).to(device)
    zeros = torch.zeros_like(real_logit).to(device)
    loss = bce_stable(real_logit - torch.mean(fake_logit), ones) + bce_stable(fake_logit - torch.mean(real_logit), zeros)
    return loss + torch.mean(real_logit).abs() #+ torch.mean(fake_logit).abs()


def RaGANLoss_g(fake_logit, real_logit):
    ones = torch.ones_like(real_logit).to(device)
    zeros = torch.zeros_like(real_logit).to(device)
    loss = bce_stable(real_logit - torch.mean(fake_logit), zeros) + bce_stable(fake_logit - torch.mean(real_logit), ones)
    return loss + torch.mean(fake_logit).abs()
    # return loss 


# fake_logit=torch.arange(12).view(6,2).float().to(device)
# real_logit=torch.arange(1,13).view(6,2).float().to(device)

# ll = RaGANLoss_g(fake_logit, real_logit)
# print(ll)

