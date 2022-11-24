import time, itertools, os

from matplotlib.pyplot import axis
from dataset import ImageFolder
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from discriminator import *
# from utils import *
from UNetDiscriminator import UNetDiscriminatorSN
from loss import *
from generator import TinyGenerator
from DCGAN import DCGenerator, DCDiscriminator
from DCGAN2 import DCGAN_G
from snnet import SNResNetProjectionDiscriminator
from GRUDiscriminator import MinibatchDiscriminator, GRUDiscriminator#, EncoderDiscriminator
from encoder_discriminator import EncoderDiscriminator, PPP, SAConv2dLayer, SSSDiscriminator, Gauss2Feature
from gngan import normalize_gradient
import adai_optim
from DiffAugment import DiffMask

import matplotlib.pyplot as plt
import numpy as np

from gan_loss import *
focal_loss = FocalLoss()

# device = 'cpu'
device = 'cuda'
batch = 2
# torch.cuda.empty_cache()

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256 + 20, 256+20)),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# train_folder = ImageFolder(os.path.join('/home/cr/Desktop', 'B_512'), train_transform)
train_folder = ImageFolder(os.path.join('/home/cr/Desktop', 'B_512'), train_transform)
train_loader = DataLoader(train_folder, batch_size=batch, shuffle=True, drop_last=True)
train_iter = iter(train_loader)


""" define generator and discriminator """
# gen = TinyGenerator().to(device)
gen = DCGenerator(512, [512, 256, 128, 64, 64, 64], 3).to(device)
gauss2f = Gauss2Feature(128, 512).to(device)
gaussdis = SSSDiscriminator(512, spectral_norm=False).to(device)
# gen = DCGAN_G().to(device)

# dis = Discriminator(input_nc=3, ndf=64, n_layers=7).to(device)
# dis = Discriminator(input_nc=3, ndf=32, n_layers=5).to(device)
# dis = DCDiscriminator(3, [64, 128, 256, 512], 1).to(device)
# dis = SNResNetProjectionDiscriminator(32).to(device)
# dis = UNetDiscriminatorSN(3, 32).to(device)
# dis = MinibatchDiscriminator(256, channel_multiplier=1).to(device)
# dis = GRUDiscriminator(256, channel_multiplier=1).to(device)
# dis = EncoderDiscriminator(256, channel_multiplier=1).to(device)
# dis = EncoderDiscriminator().to(device)
dis = PPP().to(device)



""" optimizer """
G_optim = torch.optim.AdamW(itertools.chain(gen.parameters(), gauss2f.parameters()), lr=0.0002, betas=(0.5, 0.99))
D_optim = torch.optim.AdamW(itertools.chain(dis.parameters(), gaussdis.parameters()), lr=0.0002, betas=(0.5, 0.99))



""" load checkpoint """
file_exist = os.path.exists(os.path.join("./result/checkpoint", 'ck.pt'))
if (file_exist):
    print("load checkpoint")
    params = torch.load(os.path.join("./result/checkpoint", 'ck.pt'))
    gen.load_state_dict(params['gen'])
    dis.load_state_dict(params['dis'])
    # dis.dis = SSSDiscriminator(512, spectral_norm=True).to(device)
    gauss2f.load_state_dict(params['gauss2f'])
    gaussdis.load_state_dict(params['gaussdis'])
    

style1 = torch.Tensor([[ 0.2889,  0.9718, -0.2529,  0.9238,  0.0190, -0.0152,  0.4187, -1.0580,
         -0.6235,  1.0628, -0.6498,  0.3649, -0.4502, -2.2582, -1.3992, -0.1565,
          0.3426,  1.3767, -0.3078, -1.1222,  1.8611, -1.7361, -0.2843,  1.2200,
         -1.3063,  1.6459, -0.6834,  2.4231, -1.2896,  1.3538,  0.6389,  2.3000,
          0.5666, -2.7979,  0.9048,  0.1412,  0.6885, -1.0801,  1.2177,  0.2850,
          0.1942,  0.7229, -0.2607, -0.7256, -0.3125, -1.1288,  0.4982, -0.8873,
          0.7161, -0.9434,  0.7226,  0.6055, -1.0502, -1.5978, -0.8100, -0.8720,
         -0.1642, -0.0067, -1.0826,  0.8772, -0.5285, -0.2653, -0.0603,  1.2692,
         -0.3859, -0.7247,  0.7848,  0.2746,  0.4271, -0.2447,  0.1034,  2.4866,
          0.3790, -0.0766,  1.8732, -0.3895, -1.7854, -2.2327, -2.4139,  0.2357,
         -0.7857, -0.7286,  1.1900, -0.3987,  0.3076, -0.3980,  0.6111,  0.2288,
          1.2721, -0.9279, -1.7191,  0.2612,  0.7016, -1.5784,  2.5097,  0.0517,
          0.7644,  0.0947,  0.1499,  1.7157, -0.5629,  0.9879, -0.8308, -1.2680,
          0.4060,  0.9325, -0.3055,  0.6870,  0.8560,  0.7449,  0.3312,  0.2253,
          0.8902,  1.2615, -0.5894,  0.4590, -0.2613, -0.7865,  0.9924, -0.6888,
          0.2980, -1.5313,  0.2006,  1.2643,  1.1696, -0.1505,  0.7454, -0.7921]])
style2 = torch.Tensor([[-0.8658, -1.4750,  1.2047, -2.8967, -1.3652, -0.4652,  1.4319,  0.7967,
          1.8950,  0.5507, -1.0949, -0.2681, -0.0288,  0.5419,  1.1970, -1.3234,
          0.6857, -0.6791, -1.1386,  0.5902, -0.3134,  0.1646,  0.6715,  0.6172,
          1.2187, -1.8421,  0.7773,  0.0863, -0.1701,  1.3759,  0.0783,  0.0999,
         -1.0189, -0.1556, -0.8824, -1.1971,  0.6402, -0.5589, -0.5834,  0.8715,
         -0.0586, -0.3436,  0.1114,  0.2358,  0.3463,  0.3388, -0.2201, -0.0850,
          0.1632,  0.2341, -0.8525, -1.1730,  1.0916, -1.5059, -0.4007, -1.0010,
          0.1800,  0.6150, -1.1353, -0.4540, -1.0239,  0.2965, -0.1547,  0.0347,
         -2.2598, -0.1336, -0.2628,  1.8032,  0.3039, -0.2087, -0.5589,  1.9517,
          0.0795, -0.5240, -1.9250, -0.1441,  0.3080,  0.8524,  0.7051,  0.0618,
          0.9345, -0.1937, -0.9101,  1.2158,  0.4337, -0.6616, -1.2781,  1.1225,
         -0.7798,  0.4289,  0.1634,  1.5183, -0.7094,  0.2081,  0.6017,  1.4129,
          2.2118, -0.1970,  0.0563,  0.3483,  0.1742, -2.1034,  0.7355, -0.7900,
          0.6353, -0.2419, -1.4729, -1.9154,  0.1761,  0.4780, -0.4580,  0.8413,
         -0.2074,  0.1854,  0.5829, -0.1228,  1.4155, -0.2938,  0.0532, -0.0593,
          0.0176, -0.1095,  0.0425, -0.9515, -0.2282, -1.3220, -0.6079, -0.3814]])


# mean_r1_real = torch.randn(2, 512).to(device)
# mean_r1_fake = torch.randn(2, 512).to(device)
train_d = True
train_g = True
rsign_fake = 0
rsign_real = 0
kkkkk = 7.

fake_rate = 0.01
# fake_rate = 0.47
feature_rate = 0.01
# feature_rate = 10.
reg_k = 40.
real_positive_sum = 0
fake_negtive_sum = 0
real_positive_sum2 = 0
fake_negtive_sum2 = 0

rfeature10 = None

for step in range(0, 1000000):
# for step in range(0, 30):
    gen.train(), dis.train(), gauss2f.train(), gaussdis.train()
    try:
        real_img, _ = train_iter.next()
    except:
        train_iter = iter(train_loader)
        real_img, _ = train_iter.next()
    real_img = real_img.to(device)

    # real samples
    gauss_sample_4real = torch.randn(real_img.shape[0], 128).to(device)
    fake_sample_4real, _ = gen(gauss2f(gauss_sample_4real))
    real_index = (torch.rand(real_img.shape[0], 1, 1, 1).to(device)-fake_rate).ceil()
    real_sample = (real_img*real_index + fake_sample_4real.detach()*(1.-real_index)).detach_()
    real_sample.requires_grad = True
    # real_sample = real_img
    # real_sample.requires_grad = True
    
    # fake samples
    random_styles = torch.randn(real_img.shape[0], 128).to(device)
    feature = gauss2f(random_styles)
    fake_sample = gen(feature)[0].detach_()
    
    # train D
    real_logit, _ = dis(real_sample)
    _, rfeature = dis(real_img)
    fake_logit, ffeature = dis(fake_sample)

    real_logit2 = gaussdis(rfeature.detach())
    fake_logit2 = gaussdis(feature.detach())

    recon_real, _ = gen(rfeature)
    _, recon_rfeature = dis(recon_real)
    recon_loss = 100.* F.mse_loss(recon_real, real_img.detach())
    

    adv1_loss = bce_logits_loss(real_logit - fake_logit, torch.ones_like(real_logit).to(device))
    adv1_loss2 = bce_logits_loss(real_logit2 - fake_logit2, torch.ones_like(real_logit).to(device))
    # adv1_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
    # adv1_loss = 1.*bce_logits_loss(real_logit - fake_logit, 1.0 - 0.2*torch.rand_like(real_logit).to(device))
    # adv1_loss = 1.*(torch.mean((real_logit - torch.mean(fake_logit) - 1.0*torch.ones_like(real_logit).to(device)) ** 2) +\
    #  torch.mean((fake_logit - torch.mean(real_logit) + 1.0*torch.ones_like(real_logit).to(device)) ** 2))/2
    # adv1_loss = -torch.mean(real_logit) + torch.mean(fake_logit)

    # lips = 1.*F.mse_loss(recon_rfeature, rfeature.detach())
    lips = 2.*F.mse_loss(recon_rfeature, rfeature)
    # kld = feature_rate * ((rfeature*rfeature).mean() + (ffeature*ffeature).mean())
    # kld = feature_rate * (flatent.abs().mean() + rlatent.abs().mean())

    # step1_loss = adv1_loss
    if step > 100000:
        step1_loss = adv1_loss+adv1_loss2 + recon_loss #+ lips#+ recon_loss2
    else:
        step1_loss = recon_loss + adv1_loss2 #+ lips

    if step > 100000 and step % 2 == 0:
        # D_optim.zero_grad()
        # G_optim.zero_grad()
        real_sample.requires_grad = True
        # real_logit, _ = dis(real_sample)
        # r1_loss = d_r1_loss(real_logit, real_sample)
        # noise = torch.randn_like(real_logit).to(device)
        grad_real, = torch.autograd.grad(outputs=real_logit.sum(), inputs=real_sample, create_graph=True)
        grad_real = grad_real.view(batch,-1)
        r1_loss = (grad_real*grad_real).sum(1).mean()
        real_sample.requires_grad = False
        # D_r1_loss = 5.*(5. * r1_loss * 4)
        D_r1_loss = 5*(5. * r1_loss * 4)
        # print("kkkkkk", D_r1_loss)
        step1_loss = step1_loss + D_r1_loss
        # D_r1_loss.backward(retain_graph=True)
        # D_optim.step()

    D_optim.zero_grad()
    G_optim.zero_grad()
    step1_loss.backward()
    D_optim.step()
    G_optim.step()

    if step%500==0:
        print("step: %d, adv1_loss: %.4f, adv1_loss2: %.4f, recon_loss:%.4f, lips:%.4f" % (step, adv1_loss, adv1_loss2, recon_loss, lips))



    # Update G
    fake_sample, _ = gen(gauss2f(random_styles).detach())
    real_logit, _ = dis(real_sample)
    real_logit_ori, rfeature = dis(real_img)
    fake_logit, ffeature = dis(fake_sample)

    real_logit2 = gaussdis(rfeature.detach())
    fake_logit2 = gaussdis(gauss2f(random_styles))

    recon_real, _ = gen(rfeature.detach())
    _, recon_rfeature = dis(recon_real)

    lips = 2.0*F.mse_loss(recon_rfeature, rfeature.detach())

    # adv2_loss = (bce_logits_loss(real_logit - torch.mean(fake_logit), torch.zeros_like(real_logit).to(device)) +\
    #     bce_logits_loss(fake_logit - torch.mean(real_logit), torch.ones_like(real_logit).to(device)))/2
    # adv2_loss = 1.*bce_logits_loss(fake_logit - real_logit, 1.0*torch.ones_like(real_logit).to(device))
    adv2_loss1 = bce_logits_loss(fake_logit - real_logit, torch.ones_like(real_logit).to(device))
    adv2_loss2 = bce_logits_loss(fake_logit2 - real_logit2, torch.ones_like(real_logit).to(device))
    # adv2_loss = F.softplus(-fake_logit).mean()
    # adv2_loss = 1.*bce_logits_loss(fake_logit - real_logit, 1.0 + 0.1*torch.randn_like(real_logit).to(device))
    
    # step2_loss = adv2_loss1
    recon_loss = 100.* F.mse_loss(recon_real, real_img.detach())
    if step > 100000:
        step2_loss = adv2_loss1 + adv2_loss2 + recon_loss #+ lips
    else:
        step2_loss = recon_loss + adv2_loss2 #+ lips
    G_optim.zero_grad()
    D_optim.zero_grad()
    step2_loss.backward()
    G_optim.step()

    # real_logit_ori, _, _ = dis(real_img)
    # real_logit_ori, _, _ = dis(real_sample)
    
    # real_positive_sum = real_positive_sum + real_logit_ori.sign().sum().detach()
    # fake_negtive_sum = fake_negtive_sum + fake_logit.sign().sum().detach()
    # real_positive_sum = real_positive_sum + (real_logit_ori-fake_logit).sign().sum().detach()
    fake_negtive_sum = fake_negtive_sum + (fake_logit-real_logit).sign().sum().detach()
    real_positive_sum = real_positive_sum + (real_logit_ori-fake_logit).sign().sum().detach()
    # fake_negtive_sum = fake_negtive_sum + (fake_logit-real_logit_ori).sign().sum().detach()

    fake_negtive_sum2 = fake_negtive_sum2 + (fake_logit2-real_logit2).sign().sum().detach()
    real_positive_sum2 = real_positive_sum2 + (real_logit2-fake_logit2).sign().sum().detach()

    if step % 500 == 0:
        rrrr = 0.5*real_positive_sum/(500*batch)
        ffff = 0.5*fake_negtive_sum/(500*batch)
        # rrrr = real_positive_sum/(100*batch)
        # ffff = fake_negtive_sum/(100*batch)

        rrrr2 = 0.5*real_positive_sum2/(500*batch)
        ffff2 = 0.5*fake_negtive_sum2/(500*batch)

        real_positive_sum = 0
        fake_negtive_sum = 0
        real_positive_sum2 = 0
        fake_negtive_sum2 = 0
        # if (rrrr-ffff)/2.0 > 0.5:
        #     feature_rate = feature_rate + 0.01
        # else:
        #     feature_rate = feature_rate - 0.01
        # if feature_rate < 0.01:
        #     feature_rate = 0.01
        if (rrrr-ffff)/2.0 > 0.5:
            fake_rate = fake_rate + 0.01
        else:
            fake_rate = fake_rate - 0.01
        if fake_rate < 0.01:
            fake_rate = 0.01
        # if  (rrrr+ffff) > 0.07:
        #     feature_rate = feature_rate+5*(rrrr+ffff)
        # else:
        #     feature_rate = feature_rate-0.05
        # if feature_rate<0.01:
        #     feature_rate = 0.01
        # # if (rrrr-ffff)/2.0 > 0.6 and (rrrr+ffff) > 0.1:
        # if (rrrr-ffff)/2.0 > 0.8:
        #     fake_rate = fake_rate + 0.04
        # elif (rrrr-ffff)/2.0 > 0.7:
        #     fake_rate = fake_rate + 0.02
        # elif (rrrr-ffff)/2.0 > 0.6:
        # # elif rrrr > 0.6:
        #     fake_rate = fake_rate + 0.01
        # else:
        #     fake_rate = fake_rate - 0.01
        # if fake_rate < 0.01:
        #     fake_rate = 0.01
        # if fake_rate > 0.9:
        #     fake_rate = 0.9
        print("step: %d, adv2_loss1: %.4f, adv2_loss2:%.4f, recon_loss:%.4f, lips:%.4f" % (step, adv2_loss1, adv2_loss2, recon_loss, lips))
        # print("kkkkkk", D_r1_loss)
        # print("real_logit:", real_logit)
        # print("fake_logit:", fake_logit)
        print("rrrr:", rrrr)
        print("ffff:", ffff)
        print("rrrr2:", rrrr2)
        print("ffff2:", ffff2)
        print("fake_rate:", fake_rate)
        print("-----------------------------------------")
    # if step % 50 == 0:
    #     print("fake_rate:", fake_rate)
    # if step % 1000 == 0:
    #     for m in dis.modules():
    #         if isinstance(m, SAConv2dLayer):
    #             m.get_new_kernels()

    if step % 200 == 0:
        gen.eval(), dis.eval(), gauss2f.eval()
        test_result = []
        # for i in range(10):
        randn_styles = torch.randn(2, 128).to(device)
        randn_styles[0] = style1.to(device)
        # _, fake_style = dis(fake_img.detach())
        _, real_fff = dis(real_img.detach())
        # real_style_norm = standard_normalize(real_style)
        real_style_norm = torch.randn(2, 128).to(device)
        
        real_img1 = DiffMask(real_img[0:1,:])
        # real_img1 = F.upsample(real_img1, scale_factor=0.5, mode='bicubic')
        # real_img1 = F.upsample(real_img1, scale_factor=0.5, mode='bicubic')
        # real_img1 = F.upsample(real_img1, scale_factor=0.5, mode='bicubic')
        # real_img1 = F.upsample(real_img1, scale_factor=0.5, mode='bicubic')
        # real_img1 = F.upsample(real_img1, scale_factor=16, mode='nearest')
        real_img2 = real_img[1:2,:]
        # real_img2 = F.upsample(real_img2, scale_factor=0.5, mode='bicubic')
        # real_img2 = F.upsample(real_img2, scale_factor=0.5, mode='bicubic')
        # real_img2 = F.upsample(real_img2, scale_factor=0.5, mode='bicubic')
        # real_img2 = F.upsample(real_img2, scale_factor=0.5, mode='bicubic')
        # real_img2 = F.upsample(real_img2, scale_factor=16, mode='nearest')
        # if i == 0:
        #     styles = style1.to(device)
        # if i == 1:
        #     styles = style2.to(device)
        rand_img, _ = gen(gauss2f(randn_styles))
        recon_img1, _ = gen(real_fff)
        kktt = recon_img1[0:1,:]
        # kktt = F.upsample(kktt, scale_factor=0.5, mode='bicubic')
        # kktt = F.upsample(kktt, scale_factor=0.5, mode='bicubic')
        # kktt = F.upsample(kktt, scale_factor=0.5, mode='bicubic')
        # kktt = F.upsample(kktt, scale_factor=0.5, mode='bicubic')
        # kktt = F.upsample(kktt, scale_factor=4, mode='nearest')

        col = [real_img1, kktt]
        col = torch.cat(col, 2)
        test_result.append(col)
        col = [real_img2, recon_img1[1:2,:]]
        col = torch.cat(col, 2)
        test_result.append(col)
        col = [rand_img[0:1,:], rand_img[1:2,:]]
        col = torch.cat(col, 2)
        test_result.append(col)

        test_result = torch.cat(test_result, 0).detach().cpu()
        os.makedirs('result', exist_ok=True)
        utils.save_image(test_result, os.path.join('result', 'step_%07d.png' % step), normalize=True, range=(-1, 1), nrow=16) 

        params = {}
        params['gen'] = gen.state_dict()
        params['dis'] = dis.state_dict()
        params['gauss2f'] = gauss2f.state_dict()
        params['gaussdis'] = gaussdis.state_dict()
        os.makedirs('result/checkpoint', exist_ok=True)
        torch.save(params, os.path.join("./result/checkpoint", 'ck.pt'))        

    if step % 500 == 0:
        gen.eval(), dis.eval(), gauss2f.eval()
        plt.figure("figure")
        plt.subplot(2,1,1), plt.title('real')
        if rfeature10 == None:
            rfeature10 = rfeature.detach()
        else:
            rfeature10 = torch.cat([rfeature10, rfeature.detach()],dim=0)
        rfeature_data = rfeature10.detach().cpu().numpy()
        for i in range(rfeature10.shape[0]):
            plt.scatter(np.arange(0, 512)/10.,rfeature_data[i],alpha=0.2)
        if rfeature10.shape[0] == 10:
            rfeature10 = rfeature10[2:10, :]

        plt.subplot(2,1,2), plt.title('fake')
        # plt.gca().set_aspect(1)
        gauss = torch.randn(10, 128).to(device)
        kkkk = gauss2f(gauss).detach().cpu().numpy()
        for i in range(10):
            plt.scatter(np.arange(0, 512)/10.,kkkk[i],alpha=0.2)

        # save img
        os.makedirs('result', exist_ok=True)
        plt.savefig("./result/step_%07d_fff.png" % step)
        plt.clf()