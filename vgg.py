import torch
import torch.nn as nn
VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, VGG_type, in_channels, num_classes):
        super(VGG, self).__init__()
        self.VGG_type = VGG_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(self.VGG_type)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.fcs = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
    
    def create_conv_layers(self, VGG_type):
        architect = VGG_types[VGG_type]
        layers = []
        in_channels = self.in_channels
        for x in architect:
            if type(x) == int:
                layers += [nn.Sequential(
                nn.Conv2d(in_channels, x, 3, 1, 1),
                nn.BatchNorm2d(x),
                nn.ReLU(),)]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(2,2)]
        return nn.Sequential(*layers)