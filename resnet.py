import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in_channels = 64
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(layers[0], 64, stride=1)
        self.layer2 = self._make_layer(layers[1], 128, stride=2)
        self.layer3 = self._make_layer(layers[2], 256, stride=2)
        self.layer4 = self._make_layer(layers[3], 512, stride=2)
        
        self.adap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
#         print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
#         print(x.shape)
        x = self.layer1(x)
#         print(x.shape)
        x = self.layer2(x)
#         print(x.shape)
        x = self.layer3(x)
#         print(x.shape)
        x = self.layer4(x)
#         print(x.shape)
        x = self.adap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        
    def _make_layer(self, num_layer, intermediate_channels, stride=1):
        layers = []
        identity_downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, intermediate_channels*4, 1, stride=stride, bias=False),
            nn.BatchNorm2d(intermediate_channels*4),
            
        )
        
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels*4
        for i in range(num_layer-1):
            layers.append(block(self.in_channels, intermediate_channels))
            
        return nn.Sequential(*layers)

class block(nn.Module):
    def __init__(self, in_channels, intermemiate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermemiate_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermemiate_channels)
        self.conv2 = nn.Conv2d(intermemiate_channels, intermemiate_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermemiate_channels)
        self.conv3 = nn.Conv2d(intermemiate_channels, intermemiate_channels*self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermemiate_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    def forward(self, x):
#         print(x.shape)
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
#         print(x.shape)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.relu(x)
        return x

def ResNet50(img_channel=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet([3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet([3, 8, 36, 3], img_channel, num_classes)