import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.upsample(x)
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        out = self.conv6(x)
        return out
    
class GlobalDilatedCNN(nn.Module):
    def __init__(self):
        super(GlobalDilatedCNN, self).__init__()
        # Initial convolution: no dilation, basic feature extraction.
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1, dilation=1)
        # Stacked dilated convolutions to expand the receptive field.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=16, dilation=16)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=32, dilation=32)
        # Final convolution to produce a single-channel output.
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))  # Receptive field: 3x3
        x = self.relu(self.conv2(x))  # Receptive field: 7x7  (3 + 2*(3-1)=7)
        x = self.relu(self.conv3(x))  # Receptive field: 15x15 (7 + 2*(3-1)*2=15)
        x = self.relu(self.conv4(x))  # Receptive field: 31x31 (15 + 2*(3-1)*4=31)
        x = self.relu(self.conv5(x))  # Receptive field: 63x63 (31 + 2*(3-1)*8=63)
        x = self.relu(self.conv6(x))  # Receptive field: 127x127 (63 + 2*(3-1)*16=127)
        out = self.conv_out(x)
        return out
    
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        # Initial convolution: no dilation, basic feature extraction.
        self.linear1 = nn.Linear(100, 64)
        # Stacked dilated convolutions to expand the receptive field.
        self.linear2 = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64,10000)
        
    def forward(self, x):
        x = torch.masked_select(x[:, :, :, 0], x[:, :, :, 1]).reshape(-1, 100)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.tanh(x)
        x = self.linear4(x)
        x = self.tanh(x)
        return x.reshape(-1, 100, 100)
    
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        # Initial convolution: no dilation, basic feature extraction.
        self.linear1 = nn.Linear(100, 64)
        # Stacked dilated convolutions to expand the receptive field.
        self.linear2 = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64,10000)
        
    def forward(self, x):
        x = torch.masked_select(x[:, :, :, 0], x[:, :, :, 1]).reshape(-1, 100)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.tanh(x)
        x = self.linear4(x)
        x = self.tanh(x)
        return x.reshape(-1, 100, 100)
    
