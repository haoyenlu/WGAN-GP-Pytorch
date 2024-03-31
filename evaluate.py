import torch
import argparse
import random
import numpy as np
import torchvision
import os
import torchvision.transforms as transforms
from torchvision.models.inception import Inception3

import math
from utils import get_cifar_dataset
from models import WGAN_GP, GoogLeNet


def train_inceptionV3(train_loader,test_loader,epochs=20,save_path="./inception_save"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoogLeNet()

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,weight_decay=0.01)
    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss, total, correct = 0 , 0 ,0
        for inputs, labels in train_loader:
            inputs,labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            train_loss += loss.item()

            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        train_acc = correct / total 
        train_loss = train_loss / len(train_loader)

        # testing step
        test_loss,total,correct = 0 , 0 , 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs,labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = correct / total
        test_loss = test_loss / len(test_loader)

        torch.cuda.empty_cache()

        print(f"Iteration:{epoch+1}/{epochs},   Train Loss:{round(train_loss,4)},   Train Acc %:{round(train_acc * 100,4)},  Test Loss:{round(test_loss,4)}, Test Acc %:{round(test_acc * 100,4)}")
    
        if test_acc > best_test_acc:
            torch.save(model.state_dict(),save_path)
            print("Model saved in file: {}".format(save_path))
            best_test_acc = test_acc

    return model
    


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def calculate_inception_score(images,model,n_split=10,eps=1e-16):
    model.eval()
    yhat = model(images).detach().numpy() 

    kl_list = list()
    num_images, num_classes = yhat.shape

    for i in range(num_images):
        p_yx = [sigmoid(yhat[i,j]) for j in range(num_classes)] # the output probability of the Inception network
        p_y = np.mean(p_yx) #  the probability summing all the classes

        kl_d = 0
        for i in range(num_classes):
            kl_d += p_yx[i]*(np.log(p_yx[i] + eps) - np.log(p_y+eps))
        
        kl_list.append(kl_d)


    return np.exp(np.mean(kl_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--G_ckpt','-g',help="Generator checkpoint",default=None)
    parser.add_argument('--D_ckpt','-d',help="Discriminator checkpoint",default=None)
    parser.add_argument('--sample_size',type=int,help="number of sample images",default=100)
    parser.add_argument('--g_channel',type=int,help="Generator Residual block channel",default=256)
    parser.add_argument('--use_sm',action="store_true",help="Use self-modulate generator")
    parser.add_argument('--save_pth',help="Path to image saving folder",default="./samples")
    parser.add_argument('--save_name',help="Sample image save name",default="sample.png")
    parser.add_argument('--train_inception',action="store_true",help="Train inception")
    args = parser.parse_args()

    os.makedirs(args.save_pth,exist_ok=True)
    incep_model_save = "./inception_model"
    incep_model_path = "v3.pth"
    os.makedirs(incep_model_save,exist_ok=True)

    if args.train_inception:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=2)
        model = train_inceptionV3(train_loader,test_loader,save_path=os.path.join(incep_model_save,incep_model_path))

    else:
        ckpt = torch.load(os.path.join(incep_model_save,incep_model_path))
        model = GoogLeNet()
        model.load_state_dict(ckpt)

    wgan_gp = WGAN_GP(g_channel=args.g_channel,batch_size=128,G_checkpoint=args.G_ckpt,D_checkpoint=args.D_ckpt,use_sm=args.use_sm)
    fake_images = wgan_gp.generate_samples(args.sample_size)
    grid = torchvision.utils.make_grid(fake_images,nrow=10)

    torchvision.utils.save_image(grid,f"{args.save_pth}/{args.save_name}")

    
    model.to(torch.device('cpu'))
    score = calculate_inception_score(fake_images,model)
    print(f"Inception score:{score}")

