
import time, os
import logging
from logger import setup_logger

from models.paddle2torch import *
from loss import bce_logits_loss, mse_loss, RaGANLoss_d, RaGANLoss_g, RpGANLoss_d, RpGANLoss_g, Standardization, FocalRpGANLoss_d
from visual import save_image, NCHW2HWC

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

style1 = torch.to_tensor([[ 0.2889,  0.9718, -0.2529,  0.9238,  0.0190, -0.0152,  0.4187, -1.0580,
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

class Trainer:
    """An epoch-based trainer.
    Example::
        # create your model / optimizer / lr_scheduler / data_loader before using the trainer
        model = ...
        optimizer = ...
        lr_scheduler = ...
        data_loader = ...
        # train 100 epochs
        trainer = Trainer(model, optimizer, lr_scheduler, data_loader, max_epochs=100)
        trainer.train()
    """

    def __init__(
        self,
        args,
        encoder,
        decoder,
        img_discriminator,
        feature_discriminator,
        feature_generator,
        G_optim,
        D_optim,
        data_loader
    ):
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.img_discriminator = img_discriminator
        self.feature_discriminator = feature_discriminator
        self.feature_generator = feature_generator
        self.G_optim = G_optim
        self.D_optim = D_optim
        self.data_loader = data_loader

        self._data_iter = iter(data_loader)
        self.epoch = 0
        self.start_epoch = 0
        self.inner_iter = 0
        self.real_slide_window = None
        self.fake_slide_window = None
        self.fake_rate = 0.01

        self.logger = logging.getLogger(__name__)
        setup_logger()

    def train_one_iter(self) -> None:
        iter_start_time = time.perf_counter()
        self.encoder.train()
        self.decoder.train()
        self.img_discriminator.train()
        self.feature_discriminator.train()
        self.feature_generator.train()

        try:
            real_img = next(self._data_iter)[0]
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            real_img = next(self._data_iter)[0]
        gauss_sample_4real = torch.randn([real_img.shape[0], 512])
        fake_sample_4real = self.decoder(gauss_sample_4real)
        real_index = (torch.rand([real_img.shape[0], 1, 1, 1])-self.fake_rate).ceil()
        real_sample = real_img*real_index + fake_sample_4real.detach()*(1.-real_index)
        real_sample.stop_gradient = False

        random_gauss = torch.randn([real_sample.shape[0], 512])
        fake_sample = self.decoder(random_gauss)
        
        # train D
        real_feature = self.encoder(real_sample)
        fake_feature = self.encoder(fake_sample.detach())
        recon_real = self.decoder(Standardization(real_feature))
        recon_feature = self.encoder(recon_real.detach())
        recon_ff = Standardization(self.encoder(fake_sample))

        # real_logit = real_feature.norm(2,1)/512.
        # fake_logit = fake_feature.norm(2,1)/512.
        real_logit = real_feature.mean(1)
        fake_logit = fake_feature.mean(1)
        recon_logit = recon_feature.mean(1)

        if self.inner_iter % 200 == 0:
            print("real_logit", real_logit)
            print("fake_logit", fake_logit)

        recon_loss = 10.* mse_loss(recon_real, real_sample.detach())+ 10.* mse_loss(recon_ff, random_gauss)
        center_loss = (real_feature.mean(1) - real_feature[torch.randperm(real_feature.shape[0])].mean(1)).abs().mean() + (fake_feature.mean(1) - fake_feature[torch.randperm(fake_feature.shape[0])].mean(1)).abs().mean()
        std_loss = real_feature.std(1).mean() #+ real_feature.std(1).mean()

        adv_loss1 = RpGANLoss_d(fake_logit, real_logit) + RpGANLoss_d(recon_logit, real_logit)

        step1_loss = adv_loss1 + recon_loss #+ 0.1*center_loss + 0.1*std_loss

        # if self.inner_iter % 4 == 0:
        #     grad_real, = torch.grad(outputs=real_logit.sum(), inputs=real_sample,retain_graph=True, create_graph=True)
        #     grad_real = grad_real.view([real_sample.shape[0],-1])
        #     grad_real_norm = grad_real.norm(2, 1)
        #     grad_fake, = torch.grad(outputs=fake_logit.sum(), inputs=fake_sample,retain_graph=True, create_graph=True)
        #     grad_fake = grad_fake.view([real_sample.shape[0],-1])
        #     grad_fake_norm = grad_fake.norm(2, 1)
        #     r1_loss = ((grad_real_norm-grad_fake_norm)**2).mean()
        #     step1_loss = step1_loss + 10000*r1_loss
    
        self.D_optim.zero_grad()
        self.G_optim.zero_grad()
        step1_loss.backward()#retain_graph=True
        self.D_optim.step()
        self.G_optim.step()

        # train G
        fake_sample = self.decoder(random_gauss)
        real_feature = self.encoder(real_sample)
        fake_feature = self.encoder(fake_sample)
        recon_ff = Standardization(self.encoder(fake_sample))
        recon_real = self.decoder(Standardization(real_feature))
        recon_feature = self.encoder(recon_real)

        # real_logit = real_feature.norm(2,1)/512.
        # fake_logit = fake_feature.norm(2,1)/512.
        real_logit = real_feature.mean(1)
        fake_logit = fake_feature.mean(1)
        recon_logit = recon_feature.mean(1)

        recon_loss = 10.* mse_loss(recon_real, real_sample.detach()) + 10.* mse_loss(recon_ff, random_gauss)

        adv_loss2 = RpGANLoss_g(fake_logit, real_logit) + RpGANLoss_g(recon_logit, real_logit)
        step2_loss = adv_loss2 + recon_loss#+ 1.* recon_loss #+ f2_loss#+ 10*f2_loss #+ recon_loss

        self.G_optim.zero_grad()
        step2_loss.backward()
        self.G_optim.step()

        real_logit_ori = self.encoder(real_img).mean(1)
        real_sign = (real_logit_ori-fake_logit).sign().view([-1]).detach()
        fake_sign = (fake_logit-real_logit).sign().view([-1]).detach()
        # real_sign = (real_feature-fake_feature).mean(1).sign().view([-1]).detach()
        # fake_sign = (fake_feature-real_feature).mean(1).sign().view([-1]).detach()
        # fake_sign = (fake_logit2-real_logit).sign().view([-1]).detach()
        if self.real_slide_window is not None:
            if (self.real_slide_window.shape[0] < 4000):
                self.real_slide_window = torch.concat([self.real_slide_window, real_sign])
                self.fake_slide_window = torch.concat([self.fake_slide_window, fake_sign])
            else:
                self.real_slide_window = torch.concat([self.real_slide_window[real_sign.shape[0]:], real_sign])
                self.fake_slide_window = torch.concat([self.fake_slide_window[fake_sign.shape[0]:], fake_sign])
        else:
            self.real_slide_window = real_sign
            self.fake_slide_window = fake_sign
        if self.real_slide_window.shape[0] == 4000:
            rrrr = self.real_slide_window.sum()/4000.
            ffff = self.fake_slide_window.sum()/4000.
            # if self.inner_iter%100==0 and (rrrr-ffff)/2.0 > 0.6:
            #     self.fake_rate = self.fake_rate + 0.01
            # if self.inner_iter%100==0 and (rrrr-ffff)/2.0 < 0.6:
            #     self.fake_rate = self.fake_rate - 0.01
            # if self.fake_rate < 0.01:
            #     self.fake_rate = 0.01
            if self.inner_iter % 200 == 0:
                self.logger.info(f"rrrr:{rrrr}, ffff:{ffff}, fake_rate:{self.fake_rate}")

        iter_end_time = time.perf_counter() - iter_start_time
        if self.inner_iter % 200 == 0:
            self.show_result(real_sample)
        if self.inner_iter % 200 == 0:
            self.logger.info(f"epoch:{self.epoch}, inner_iter:{self.inner_iter}, iter_end_time:{iter_end_time}")
            self.logger.info(f"recon_loss:{recon_loss}, adv_loss1:{adv_loss1}, adv_loss2:{adv_loss2}")
            self.logger.info("----------------------------------")        

    def train_one_epoch(self) -> None:
        for self.inner_iter in range(len(self.data_loader)):
            self.train_one_iter()

    def train(self, checkpoint_path = None) -> None:
        if checkpoint_path is not None:
            self.load_checkpoint(path=checkpoint_path)
        self.logger.info(f"Start training from epoch {self.start_epoch}")
        for self.epoch in range(self.start_epoch, self.args.max_epochs):
            if self.epoch > 0:
                self.save_checkpoint()
            self.train_one_epoch()

    def show_result(self, real_img) -> None:
        self.encoder.eval()
        self.decoder.eval()
        self.img_discriminator.eval()
        self.feature_discriminator.eval()
        self.feature_generator.eval()

        test_result = []
        randn_styles = torch.randn([4, 128])
        randn_styles[0] = style1
        randn_styles2 = torch.randn([4, 512])
        # fake_img = self.decoder(self.feature_generator(randn_styles))
        fake_img = self.decoder(randn_styles2)
        rfeature = self.encoder(real_img.detach())
        recon_img = self.decoder(Standardization(rfeature))
        recon_ff = self.encoder(fake_img)
        recon_rf = self.encoder(recon_img)

        col = [real_img[0:1,:], recon_img[0:1,:]]
        col = torch.concat(col, 2)
        test_result.append(col)
        col = [real_img[1:2,:], recon_img[1:2,:]]
        col = torch.concat(col, 2)
        test_result.append(col)
        col = [fake_img[0:1,:], fake_img[1:2,:]]
        col = torch.concat(col, 2)
        test_result.append(col)
        test_result = torch.concat(test_result, -1).detach().numpy()
        test_result = NCHW2HWC(test_result)
        os.makedirs('./result', exist_ok=True)
        save_image(test_result, os.path.join('./result', 'step_%04d_%05d.png' % (self.epoch, self.inner_iter)))

        plt.figure("figure", dpi=80)
        plt.gca().set_aspect(1)
        plt.subplot(2,3,1), plt.title('real')
        plt.gca().set_aspect(1)
        rguass = Standardization(rfeature)
        rguass = rguass.detach().numpy()
        plt.scatter(rguass[0],rguass[1],alpha=0.2)
        plt.scatter(rguass[2],rguass[3],alpha=0.2)
        plt.subplot(2,3,2), plt.title('recon_fake')
        plt.gca().set_aspect(1)
        fguass = Standardization(recon_ff)
        fguass = fguass.detach().numpy()
        plt.scatter(fguass[0],fguass[1],alpha=0.2)
        plt.scatter(fguass[2],fguass[3],alpha=0.2)
        plt.subplot(2,3,4), plt.title('rlatent')
        plt.gca().set_aspect(1)
        rlatent = rfeature.detach().numpy()
        plt.scatter(rlatent[0],rlatent[1],alpha=0.2)
        plt.scatter(rlatent[2],rlatent[3],alpha=0.2)
        plt.subplot(2,3,5), plt.title('flatent')
        plt.gca().set_aspect(1)
        flatent = recon_ff.detach().numpy()
        plt.scatter(flatent[0],flatent[1],alpha=0.2)
        plt.scatter(flatent[2],flatent[3],alpha=0.2)
        plt.subplot(2,3,6), plt.title('reconreal_latent')
        plt.gca().set_aspect(1)
        recon_rf = recon_rf.detach().numpy()
        plt.scatter(recon_rf[0],recon_rf[1],alpha=0.2)
        plt.scatter(recon_rf[2],recon_rf[3],alpha=0.2)
        plt.savefig("./result/step_%04d_%05d_gauss.png" % (self.epoch, self.inner_iter))
        plt.clf()
        

    def save_checkpoint(self, file_name=None) -> None:
        """Save training state: ``epoch``, ``num_gpus``, ``model``, ``optimizer``, ``lr_scheduler``,
        ``metric_storage``, ``hooks`` (optional), ``grad_scaler`` (optional).
        Args:
            filename (str): The checkpoint will be saved as ``ckpt_dir/filename``.
        """
        data = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "img_discriminator": self.img_discriminator.state_dict(),
            "feature_discriminator": self.feature_discriminator.state_dict(),
            "feature_generator": self.feature_generator.state_dict(),
            "G_optim": self.G_optim.state_dict(),
            "D_optim": self.D_optim.state_dict(),
        } 

        if file_name is None:
            file_name = "latest.pth"
        file_path = os.path.join(self.args.checkpoint_path, file_name)
        self.logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

    def load_checkpoint(self, path = None, auto_resume = False):
        """Load the given checkpoint or resume from the latest checkpoint.
        Args:
            path (str): Path to the checkpoint to load.
            auto_resume (bool): If True, automatically resume from the latest checkpoint.
        """
        if path is None and auto_resume:
            latest_ckpt = os.path.join(self.args.checkpoint_path, "latest.pth")
            if os.path.exists(latest_ckpt):
                path = latest_ckpt
        if path:
            self.logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path)
        else:
            self.logger.info("Skip loading checkpoint.")
            return

        self.start_epoch = checkpoint["epoch"] + 1
        self.encoder.set_state_dict(checkpoint['encoder'])
        self.decoder.set_state_dict(checkpoint['decoder'])
        self.img_discriminator.set_state_dict(checkpoint['img_discriminator'])
        self.feature_discriminator.set_state_dict(checkpoint['feature_discriminator'])
        self.feature_generator.set_state_dict(checkpoint['feature_generator'])
        self.G_optim.set_state_dict(checkpoint['G_optim'])
        self.D_optim.set_state_dict(checkpoint['D_optim'])

