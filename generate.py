import torch
import argparse
import random
import numpy as np
import torchvision

from utils import get_cifar_dataset
from models import WGAN_GP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--G_ckpt','-g',help="Generator checkpoint path",default=None)
    parser.add_argument('--D_ckpt','-d',help="Discriminator checkpoint path",default=None)
    parser.add_argument('--G_save',help="checkpoint save directory for the generator",default='./models/netG')
    parser.add_argument('--D_save',help="checkpoint save directory for the discriminator",default='./models/netD')
    parser.add_argument('--max_iter',type=int,help="Max iterations for training",default=100000)
    parser.add_argument('--batch_size',type=int,help="Batch size for training",default=256)
    parser.add_argument('--g_channel',type=int,help="The channel size for generator residual block",default=256)
    parser.add_argument('--use_sm',action="store_true",help="Use self-modulate generator")
    parser.add_argument('--use_sn',action="store_true",help="Use Spectral Normalization on discriminator")

    args = parser.parse_args()
    batch_size = args.batch_size
    g_channel = args.g_channel

    # Set random seed
    randm_seed = random.randint(1,10000)
    print("Random Seed:" ,randm_seed)
    random.seed(randm_seed)
    np.random.seed(randm_seed)
    torch.manual_seed(randm_seed)
    torch.cuda.manual_seed_all(randm_seed)

    torch.backends.cudnn.benchmark = True

    # get data
    dataloader = get_cifar_dataset(batch_size)

    # display image
    batch = next(iter(dataloader))
    print(f"Standardize image to [{torch.min(batch[0]).item()},{torch.max(batch[0]).item()}]")
    torchvision.utils.save_image(batch[0].mul(0.5).add(0.5),"images.png")

    # train model
    wgan_gp = WGAN_GP(g_channel=g_channel,batch_size=batch_size,max_iters=args.max_iter,
                      G_checkpoint=args.G_ckpt,D_checkpoint=args.D_ckpt,use_sm=args.use_sm,use_sn=args.use_sn)
    
    wgan_gp.train(dataloader,G_save_path=args.G_save,D_save_path=args.D_save)


