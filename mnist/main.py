import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

from model import LeNet, MLP



def train_process(model, trainloader, criterion, optimizer, device):
    model.train()
    all_correct = 0
    all_total = 0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs, dim=1)
        correct = (pred == labels).sum()
        total = labels.size(0)
        acc = correct / total * 100

        all_correct += correct
        all_total += total
        if i % 100 == 0:
            print(f'train acc: {acc:3.1f}, loss: {loss.item():.3f}')


@torch.no_grad()
def test_process(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(testloader):

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum()
        total += labels.size(0)
    
    acc = correct / total * 100
    print(f'-->> test acc: {acc:.6f}')
    return acc


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('ues device', device)

    batch_size = 128
    epoch_size = 200
    lr = 0.0001
    momentum = 0.9
    
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          # transforms.RandomRotation(10),
                                          transforms.Normalize((0.5), (0.5))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # net = LeNet().to(device)
    net = MLP().to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    best_acc = 0
    for epoch in range(epoch_size):

        print(f'epoch {epoch}')
        train_process(net, trainloader, criterion, optimizer, device)
        test_acc = test_process(net, testloader, device)

        if test_acc > best_acc:
            best_acc = test_acc
    print(f'final best test acc {best_acc}')
    print(f'final best test acc {best_acc:.4f}')
    print('finish!!')


if __name__ == "__main__":
    main()