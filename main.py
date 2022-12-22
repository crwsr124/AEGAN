import argparse
from config import load_cfg_from_cfg_file, merge_cfg_from_list

import time, itertools, os
import sys
sys.path.append("models")
from models.dcgan_decoder import DCGenerator
from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import BatchDiscriminator
from dataset import ImageFolder

from paddle.io import DataLoader
from paddle.vision import transforms
from models.paddle2torch import *
from train_paddle import Trainer

# paddle.device.set_device('gpu:0')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config.yaml', help='config file')
    parser.add_argument('opts', help='see config.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config_file is not None
    cfg = load_cfg_from_cfg_file(args.config_file)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def main():
    args = get_parser()
    print("batch_size:", args.batch_size)
    print("max_epochs:", args.max_epochs)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256 + 20, 256+20)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_folder = ImageFolder(args.data_root, train_transform)
    train_loader = DataLoader(train_folder, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    print("one_epoch_iters:", len(train_loader))

    encoder = Encoder(512)
    decoder = DCGenerator(512, [512, 256, 128, 64, 64, 64], 3)
    img_discriminator = BatchDiscriminator(512)
    feature_generator = Generator(128, 512)
    feature_discriminator = BatchDiscriminator(512)

    """ optimizer """
    # G_optim = torch.optim.AdamW(parameters=itertools.chain(decoder.parameters(), feature_generator.parameters()), learning_rate=0.0002, beta1=0.5, beta2=0.99)
    # D_optim = torch.optim.AdamW(parameters=itertools.chain(encoder.parameters(), img_discriminator.parameters(), feature_discriminator.parameters()), learning_rate=0.0002, beta1=0.5, beta2=0.99)
    G_optim = torch.optim.AdamW(parameters=itertools.chain(decoder.parameters(), feature_discriminator.parameters()), learning_rate=0.0002, beta1=0.5, beta2=0.99)
    D_optim = torch.optim.AdamW(parameters=itertools.chain(encoder.parameters()), learning_rate=0.0002, beta1=0.5, beta2=0.99)

    trainer = Trainer(args, encoder, decoder, img_discriminator, feature_discriminator, feature_generator, G_optim, D_optim, train_loader)
    trainer.load_checkpoint(auto_resume=args.auto_resume)
    trainer.train()

if __name__ == "__main__":
    main()



    
