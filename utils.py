import torch
import torchvision
import torchvision.transforms as transforms

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