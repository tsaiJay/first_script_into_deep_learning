import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # 2
        # self.fc1 = nn.Linear(28*28, 512)
        # self.fc2 = nn.Linear(512, 10)

        # 3
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):  # x: input
        B = x.size(0)  # batch size

        y = self.conv1(x)
        y = self.relu(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.relu(y)
        y = self.pool2(y)

        y = y.view(B, -1)  # <<<<<<<<<<<<<<<< y = y.flatten(1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        
        return y


if __name__ == "__main__":
    net = LeNet()

    fake_input = torch.randn(3, 1, 28, 28)
    print(fake_input.shape)

    fake_out = net(fake_input)
    print(fake_out.shape)