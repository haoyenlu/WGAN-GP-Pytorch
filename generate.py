import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.parametrizations import spectral_norm

from pytorch_gan_metrics import get_inception_score

import random
import math
import argparse
import numpy as np

G_ckpt_path = "./models/netG"
D_ckpt_path = "./models/netD"
sample_image_path = "./sample_images"

torch.backends.cudnn.benchmark = True



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_cifar_dataset(batch_size = 64):
    train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

    dataset = torch.utils.data.ConcatDataset([train_dataset,test_dataset])
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=4)

    return dataloader


class ResBlockUp(nn.Module):
    def __init__(self,in_channel=256,out_channel=256,kernel_size=3,padding="same"):
        super(ResBlockUp,self).__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(out_channel,out_channel,kernel_size=kernel_size,padding=padding)
        )

        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=padding)
        )

        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self,x):
        return self.residual(x) + self.shortcut(x)

class OptimizedResBlockDown(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,padding="same"):
        super(OptimizedResBlockDown,self).__init__()

        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel,out_channel,kernel_size,padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size,padding=padding),
            nn.AvgPool2d(2)
        )

        self.shortcut = nn.Sequential(
            # nn.AvgPool2d(2),
            nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=padding),
            nn.AvgPool2d(2)
        )
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                spectral_norm(m)


    def forward(self,x):
        return self.residual(x) + self.shortcut(x)
    

class ResBlockDown(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,padding="same",down=False):
        super(ResBlockDown,self).__init__()

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=kernel_size,padding=padding)
        ]
        if down: 
            residual.append(nn.AvgPool2d(2))

        shortcut = [nn.Conv2d(in_channel,out_channel,kernel_size=1,padding="same")]
        if down:
            shortcut.append(nn.AvgPool2d(2))

        self.residual = nn.Sequential(*residual)
        self.shortcut = nn.Sequential(*shortcut)

        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self,x):
        return self.residual(x) + self.shortcut(x)

 

    
class Generator(nn.Module):
    def __init__(self,z_dim=128,channel=256):
        super(Generator,self).__init__()

        self.ch = channel
        self.linear = nn.Linear(z_dim,4*4*channel)

        self.blocks = nn.Sequential(
            ResBlockUp(in_channel=channel,out_channel=channel,kernel_size=3),
            ResBlockUp(in_channel=channel,out_channel=channel,kernel_size=3),
            ResBlockUp(in_channel=channel,out_channel=channel,kernel_size=3)
        )

        self.output = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel,3,kernel_size=3,padding="same"),
            nn.Tanh()
        )
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self,z):
        out = self.linear(z)
        out = torch.reshape(out,(-1,self.ch,4,4))

        return self.output(self.blocks(out))
    
class Discriminator(nn.Module):
    def __init__(self,channel=128):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
            OptimizedResBlockDown(in_channel=3,out_channel=channel,kernel_size=3),
            ResBlockDown(in_channel=channel,out_channel=channel,kernel_size=3,down=True),
            ResBlockDown(in_channel=channel,out_channel=channel,kernel_size=3),
            ResBlockDown(in_channel=channel,out_channel=channel,kernel_size=3),
            nn.ReLU(),
        )

        self.linear = nn.Linear(channel,1,bias=False)
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        spectral_norm(self.linear)


    def forward(self,x):
        _x = self.model(x).sum(dim=[2,3])
        _x = self.linear(_x)
        return _x


class WGAN_GP:
    def __init__(self,g_channel = 256,d_channel=128,z_dim=128,max_iters=100000,batch_size=128,G_checkpoint = None,D_checkpoint=None):
        self.G = Generator(z_dim,g_channel)
        self.D = Discriminator(d_channel)

        if G_checkpoint:
            print("Load Generator checkpoint")
            ckpt = torch.load(G_checkpoint)
            self.G.load_state_dict(ckpt)
        
        if D_checkpoint:
            print("Load Discriminator checkpoint")
            ckpt = torch.load(D_checkpoint)
            self.D.load_state_dict(ckpt)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 2e-4
        self.beta1 = 0
        self.beta2 = 0.9
        self.decay_lambda = 10
        self.n_critic = 5
        self.penalty_lambda = 10


        self.g_optimizer = torch.optim.Adam(self.G.parameters(),lr=self.lr,betas=(self.beta1,self.beta2),weight_decay=self.decay_lambda)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(),lr=self.lr,betas=(self.beta1,self.beta2),weight_decay=self.decay_lambda)
        self.g_lrScheduler = torch.optim.lr_scheduler.LinearLR(self.g_optimizer,start_factor=1,end_factor=0,total_iters=max_iters)
        self.d_lrScheduler = torch.optim.lr_scheduler.LinearLR(self.d_optimizer,start_factor=1,end_factor=0,total_iters=max_iters)


        self.max_iters = max_iters
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.sample_size = 64

        self.writer = SummaryWriter()

    def train(self,dataloader):
        summary(self.G,(1024,128))
        summary(self.D,(3,32,32))

        self.data = self.get_infinite_batches(dataloader)

        self.G.to(self.device)
        self.D.to(self.device)

        fixed_noise = torch.randn(self.sample_size,self.z_dim).to(self.device)

        for g_iter in range(self.max_iters):
            for p in self.D.parameters():
                p.requires_grad = True
            
            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_loss = 0

            batch_size = self.batch_size

            self.G.train()

            for d_iter in range(self.n_critic):
                self.D.zero_grad()
                self.G.zero_grad()

                images = Variable(self.data.__next__()).to(self.device)
                batch_size = images.size(0)

                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()

                z = torch.randn(batch_size,self.z_dim).to(self.device)
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()

                grad_penalty = self.calculate_gradient_penalty(real_images=images,fake_images=fake_images,batch_size=batch_size)

                d_loss = d_loss_fake - d_loss_real + grad_penalty
                d_loss.backward()
                Wasserstein_loss = d_loss_real - d_loss_fake

                self.d_optimizer.step()
                print(f'Discriminator iteration: {d_iter}/{self.n_critic}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
            
            for p in self.D.parameters():
                p.requires_grad = False

            self.G.zero_grad()
            self.D.zero_grad()

            z = torch.randn(batch_size,self.z_dim).to(self.device)
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = - g_loss.mean()
            g_loss.backward()

            self.g_optimizer.step()

            # learning rate decay
            self.g_lrScheduler.step()
            self.d_lrScheduler.step()

            print(f'Generator iteration: {g_iter}/{self.max_iters}, g_loss: {g_loss}')


            if g_iter % 40 == 0:
                self.G.eval()
                fake_images = self.G(fixed_noise).detach().cpu()
                fake_images = fake_images.mul(0.5).add(0.5)
                grid = torchvision.utils.make_grid(fake_images,nrow=8)
                # save_image(grid,f"{sample_image_path}/fake_samples.png")

                IS, IS_std = get_inception_score(fake_images)

                self.save_model()
                self.writer.add_scalar('Loss/Wasserstein_distance',Wasserstein_loss.item(),g_iter)
                self.writer.add_scalar('Loss/Loss_D',d_loss.item(),g_iter)
                self.writer.add_scalar('Loss/Loss_G',g_loss.item(),g_iter)
                self.writer.add_scalar('Loss/Loss_D_Real',d_loss_real.item(),g_iter)
                self.writer.add_scalar('Loss/Loss_D_Fake',d_loss_fake.item(),g_iter)
                self.writer.add_image('Sample Images',grid,g_iter)
                self.writer.add_scalar('lr/lr_G',self.g_lrScheduler.get_last_lr()[0],g_iter)
                self.writer.add_scalar('lr/lr_D',self.d_lrScheduler.get_last_lr()[0],g_iter)
                self.writer.add_scalar('IS/IS',IS,g_iter)
                self.writer.add_scalar('IS/IS_std',IS_std,g_iter)

            torch.cuda.empty_cache()
        
        self.save_model()
        self.writer.close()
        print("Finished Training!")

    
    def save_model(self):
        torch.save(self.G.state_dict(),f"{G_ckpt_path}/netG_ckpt.pth")
        torch.save(self.D.state_dict(),f"{D_ckpt_path}/netD_ckpt.pth")
        print("Model saved!")

    def get_infinite_batches(self,dataloader):
        while True:
            for images, _  in dataloader:
                yield images

    def calculate_gradient_penalty(self,real_images,fake_images,batch_size):
        epsilon = torch.FloatTensor(np.random.random((batch_size,1,1,1))).to(self.device)

        interpolate = (epsilon*real_images + ((1-epsilon)*fake_images)).requires_grad_(True).to(self.device)

        prob_interpolate = self.D(interpolate)

        gradient = torch.autograd.grad(outputs=prob_interpolate,inputs=interpolate,grad_outputs=torch.ones(prob_interpolate.size(),device=self.device),create_graph=True,retain_graph=True)[0]
        gradient = gradient.view(gradient.size(0),-1)
        grad_penalty = ((gradient.norm(2,dim=1) - 1) ** 2).mean() * self.penalty_lambda

        return grad_penalty





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--G_ckpt','-g',help="Generator checkpoint",default=None)
    parser.add_argument('--D_ckpt','-d',help="Discriminator checkpoint",default=None)
    parser.add_argument('--max_iter',help="Max iterations for training",default=100000)
    parser.add_argument('--batch_size',help="Batch size for training",default=128)
    args = parser.parse_args()


    batch_size = args.batch_size

    # Set random seed
    randm_seed = random.randint(1,10000)
    print("Random Seed:" ,randm_seed)
    random.seed(randm_seed)
    np.random.seed(randm_seed)
    torch.manual_seed(randm_seed)
    torch.cuda.manual_seed_all(randm_seed)

    # get data
    dataloader = get_cifar_dataset(batch_size)

    # display image
    batch = next(iter(dataloader))
    print(f"Standardize image to [{torch.min(batch[0]).item()},{torch.max(batch[0]).item()}]")
    torchvision.utils.save_image(batch[0].mul(0.5).add(0.5),"images.png")


    # G = Generator()
    # D = Discriminator()

    # summary(G,(1024,128))
    # summary(D,(3,32,32))


    wgan_gp = WGAN_GP(g_channel=256,batch_size=batch_size,max_iters=args.max_iter,G_checkpoint=args.G_ckpt,D_checkpoint=args.D_ckpt)
    wgan_gp.train(dataloader)


