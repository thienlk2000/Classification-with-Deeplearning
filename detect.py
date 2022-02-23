import argparse
import matplotlib.pyplot as plt
import torch
import os
import torchvision.transforms as T
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('data', help='data folder contain image to detect')
parser.add_argument('label', help='label class name of data')
parser.add_argument('model_type', help='model consist alexnet, efficientnet, resnet, vgg')
parser.add_argument('model_weight', help='weight of pretrained model')

args = parser.parse_args()
model_type = args.model_type
data_dir = args.data
weights = args.model_weight
label = args.label

with open(label, 'r') as f:
    class_name = f.readlines()

num_classes = len(class_name)


translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}

if model_type == 'alexnet':
    from alexnet import AlexNet
    model = AlexNet(num_classes=num_classes)
elif model_type == 'vgg':
    from vgg import VGG
    model = VGG('VGG16', in_channels=3, num_classes=num_classes)
elif model_type == 'resnet':
    from resnet import ResNet50
    model = ResNet50(num_classes=num_classes)
elif model_type == 'efficientnet':
    from efficientNet import EfficientNet
    model = EfficientNet('b0', num_classes=num_classes)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(weights)
model.load_state_dict(torch.load(weights,map_location=torch.device(device)))
model = model.to(device)
model.eval()
file_img = [os.path.join(data_dir,file) for file in os.listdir(data_dir)]

test_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

with torch.no_grad():
    num_data = len(file_img)
    plt.figure()
    for i in range(1,num_data+1):
        img = Image.open(file_img[i-1])
        img_transformed = test_transform(img)
        img_transformed = img_transformed.unsqueeze(0)
        plt.subplot(num_data//5+1 if ((num_data % 5) != 0) else num_data//5,5,i)
        plt.imshow(T.ToTensor()(img).permute(1,2,0))
        plt.axis('off')
        score = model(img_transformed)
        # print(score)
        index = (score.argmax(dim=1)).squeeze().item()
        pred = class_name[index]
        plt.title(pred)

    plt.show()


