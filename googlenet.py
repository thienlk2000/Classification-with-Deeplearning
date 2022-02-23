import torch
import torch.nn as nn

class GoogleNet(nn.Module):
    def __init__(self, ax=False, num_classes=1000):
        super(GoogleNet, self).__init__()
        assert ax == True or ax == False
        self.ax = ax

        self.conv1 = cnn_block(3, 64, 7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = cnn_block(64, 192, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, num_classes)
        
        if ax:
            self.ax1 = InceptionAx(512, num_classes)
            self.ax2 = InceptionAx(528, num_classes)
        else:
            self.ax1 = self.ax2 = None
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        
        if self.training and self.ax:
            ax1 = self.ax1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.training and self.ax:
            ax2 = self.ax2(x)
            
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        if self.training and self.ax:
            return x, ax1, ax2
        else:
            return x
        
        
class cnn_block(nn.Module):
    def __init__(self, input_size, output_size, kernel,**kwargs):
        super(cnn_block, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel, **kwargs)
        self.batchnorm = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Inception(nn.Module):
    def __init__(self,in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception, self).__init__()
        self.branch1 = cnn_block(in_channels, out_1x1, 1)
        self.branch2 = nn.Sequential(
            cnn_block(in_channels, red_3x3, 1),
            cnn_block(red_3x3, out_3x3, 3, padding=1),)
        self.branch3 = nn.Sequential(
            cnn_block(in_channels, red_5x5, 1),
            cnn_block(red_5x5, out_5x5, 5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            cnn_block(in_channels, out_1x1pool, 1)
        )
        
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
    
class InceptionAx(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAx, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = cnn_block(in_channels, 128, 1)
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x