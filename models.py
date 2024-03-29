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

import numpy as np
import os


class SM_Layer(nn.Module): # forward the 
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.output_size = output_size

        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size,bias=False)
        self.activation = nn.ReLU()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.xavier_normal_(self.linear2.weight)

        # spectral_norm(self.linear1)
        # spectral_norm(self.linear2)
    
    
    def forward(self,z):
        _z = self.activation(self.linear1(z))
        _z = self.linear2(_z)
        return _z.view(-1,self.output_size,1,1)




class ResBlockUp_SM(nn.Module): # self-modulate batch norm
    def __init__(self,z_dim=128,in_channel=256,out_channel=256,kernel_size=3,padding="same"):
        super(ResBlockUp_SM,self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channel,affine=False)
        self.sm_gamma_1 = SM_Layer(input_size=z_dim,hidden_size=128,output_size=in_channel)
        self.sm_beta_1 = SM_Layer(input_size=z_dim,hidden_size=128,output_size=in_channel)

        self.bn2 = nn.BatchNorm2d(out_channel,affine=False)
        self.sm_gamma_2 = SM_Layer(input_size=z_dim,hidden_size=128,output_size=out_channel)
        self.sm_beta_2 = SM_Layer(input_size=z_dim,hidden_size=128,output_size=out_channel)

        self.upsample = nn.Upsample(scale_factor=2,mode="nearest")
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=kernel_size,padding=padding)

        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=padding)
        )

        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self,x,z):
        _x = self.bn1(x)
        gamma1 = self.sm_gamma_1(z)
        beta1 = self.sm_beta_1(z)
        _x = _x * gamma1 + beta1

        _x = self.activation(_x)
        _x = self.upsample(_x)
        _x = self.conv1(_x)

        _x = self.bn2(_x)
        gamma2 = self.sm_gamma_2(z)
        beta2 = self.sm_beta_2(z)
        _x = _x * gamma2 + beta2

        _x = self.activation(_x)
        _x = self.conv2(_x)

        return _x + self.shortcut(x)


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
    def __init__(self,in_channel,out_channel,kernel_size=3,padding="same",use_sn=False):
        super(OptimizedResBlockDown,self).__init__()

        self.use_sn = use_sn

        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel,out_channel,kernel_size,padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size,stride=2,padding=1),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=2,padding=1),
        )

        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                if self.use_sn: spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                if self.use_sn: spectral_norm(m)


    def forward(self,x):
        return self.residual(x) + self.shortcut(x)
    

class ResBlockDown(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,padding="same",down=False,use_sn=False):
        super(ResBlockDown,self).__init__()

        self.use_sn = use_sn

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,padding=padding),
            nn.ReLU(),
        ]
        if down: 
            residual.append(nn.Conv2d(out_channel,out_channel,kernel_size=kernel_size + 1,stride=2,padding=1))
        

        shortcut = []
        if down:
            shortcut.append(nn.Conv2d(in_channel,out_channel,kernel_size=2,stride=2,padding=1))
        else:
            shortcut.append(nn.Conv2d(in_channel,out_channel,kernel_size=1,padding="same"))

        self.residual = nn.Sequential(*residual)
        self.shortcut = nn.Sequential(*shortcut)

        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                if self.use_sn: spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                if self.use_sn: spectral_norm(m)

    def forward(self,x):
        return self.residual(x) + self.shortcut(x)

 
class SMGenerator(nn.Module):
    def __init__(self,z_dim=128,channel=256):
        super(SMGenerator,self).__init__()

        self.ch = channel
        self.linear = nn.Linear(z_dim,4*4*channel)

        self.block1 = ResBlockUp_SM(z_dim=z_dim,in_channel=channel,out_channel=channel,kernel_size=3)
        self.block2 = ResBlockUp_SM(z_dim=z_dim,in_channel=channel,out_channel=channel,kernel_size=3)
        self.block3 = ResBlockUp_SM(z_dim=z_dim,in_channel=channel,out_channel=channel,kernel_size=3)

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
        out = self.block1(out,z)
        out = self.block2(out,z)
        out = self.block3(out,z)

        return self.output(out)
    
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
    def __init__(self,channel=128,use_sn=False):
        super(Discriminator,self).__init__()

        self.use_sn = use_sn

        self.model = nn.Sequential(
            OptimizedResBlockDown(in_channel=3,out_channel=channel,kernel_size=3,use_sn=use_sn),
            ResBlockDown(in_channel=channel,out_channel=channel,kernel_size=3,down=True,use_sn=use_sn),
            ResBlockDown(in_channel=channel,out_channel=channel,kernel_size=3,use_sn=use_sn),
            ResBlockDown(in_channel=channel,out_channel=channel,kernel_size=3,use_sn=use_sn),
            nn.ReLU(),
        )

        self.linear = nn.Linear(channel,1)


    def forward(self,x):
        _x = self.model(x).sum(dim=[2,3])
        _x = self.linear(_x)
        return _x


class WGAN_GP:
    def __init__(self,g_channel = 256,d_channel=128,z_dim=128,max_iters=100000,batch_size=128,G_checkpoint = None,D_checkpoint=None,use_sm=False,use_sn=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_sm: self.G = SMGenerator(z_dim,g_channel)
        else: self.G = Generator(z_dim,g_channel)

        self.D = Discriminator(d_channel,use_sn=use_sn)

        self.G.to(self.device)
        self.D.to(self.device)

        if G_checkpoint:
            print("Load Generator checkpoint")
            ckpt = torch.load(G_checkpoint)
            self.G.load_state_dict(ckpt)
        
        if D_checkpoint:
            print("Load Discriminator checkpoint")
            ckpt = torch.load(D_checkpoint)
            self.D.load_state_dict(ckpt)




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
        self.g_channel = g_channel
        self.sample_size = 64

        self.writer = SummaryWriter()

    def train(self,dataloader,G_save_path="./models/netG",D_save_path="./models/netD"):
        summary(self.G,(self.batch_size,128))
        summary(self.D,(3,32,32))

        os.makedirs(G_save_path,exist_ok=True)
        os.makedirs(D_save_path,exist_ok=True)

        self.data = self.get_infinite_batches(dataloader)


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

                self.save_model(G_save_path,D_save_path)
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
        
        self.save_model(G_save_path,D_save_path)
        self.writer.close()
        print("Finished Training!")

    
    def save_model(self,G_save_path,D_save_path):
        torch.save(self.G.state_dict(),f"{G_save_path}/netG_{self.g_channel}_ckpt.pth")
        torch.save(self.D.state_dict(),f"{D_save_path}/netD_{self.g_channel}_ckpt.pth")
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
    
    def generate_samples(self,sample_size):
        self.G.eval()
        noise = torch.randn(sample_size,self.z_dim).to(self.device)
        fake_images = self.G(noise).detach().cpu().mul(0.5).add(0.5)

        return fake_images






class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x