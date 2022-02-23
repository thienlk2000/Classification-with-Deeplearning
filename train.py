import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from dataset import AnimalDataset
from alexnet import AlexNet
from vgg import VGG
from resnet import ResNet50
from efficientNet import EfficientNet
from utils import split_train_val, plot_images, plot_training_process
from torch.utils.data import Dataset, DataLoader

def train(model, optimizer, scheduler,train_loader, val_loader,model_name, epoch=1, print_every=100, ):
    model = model.to(device)
    loss_iter = 0
    acc_iter = 0
    num_iter = 0
    num_loop = 0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    batch_size = train_loader.batch_size
    num_train = len(train_loader.dataset)
    for e in range(epoch):
        for i,(x, y) in enumerate(train_loader):
            # print("Step1")
            num_loop += 1
            model.train()
            x = x.to(device)
            y = y.to(device)
            score = model(x)
            loss = F.cross_entropy(score, y)
            optimizer.zero_grad()
            
            loss.backward()
            # print(model.classifier[6].weights.grad)
            optimizer.step()
            with torch.no_grad():
                acc_iter += (score.argmax(dim=1) == y).sum()
                num_iter += y.shape[0]
            loss_iter += loss.item()
            if i % print_every == print_every - 1 or i == len(train_loader) - 1:
                # print("Step2")
                # if i % print_every == print_every - 1:
                #     loss_iter /= print_every
                # else:
                #     loss_iter /=  (num_train - (len(train_loader) - 1)*batch_size)
                loss_iter /= num_loop
#             acc_train = float(acc_iter) / num_iter
                train_loss.append(loss_iter)
                train_acc.append(float(acc_iter) / num_iter)
                loss_val, acc_val = check_val_loader(model, val_loader)
                # print("Step3")
                val_loss.append(loss_val)
                val_acc.append(acc_val)
                print(f"Epoch:{e}\tIteration:{i}\tTrain Loss:{loss_iter}\tTrain Accuracy:{float(acc_iter) / num_iter}")
                print(f"Epoch:{e}\tIteration:{i}\tVal Loss:{loss_val}\tVal Accuracy:{acc_val}")
                loss_iter = 0
                acc_iter = 0
                num_iter = 0
                num_loop = 0
                if acc_val > best_acc:
                  best_acc = acc_val
                  torch.save(model.state_dict(), model_name)


        scheduler.step()
    return train_loss, train_acc, val_loss, val_acc

def check_val_loader(model, val_loader):
    loss_val = 0
    acc_val = 0
    num = 0
    model.eval()  
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device=device)
            y = y.to(device=device)
            score = model(x)
            loss = F.cross_entropy(score, y)
            loss_val += loss.item()
            acc_val += (score.argmax(dim=1) == y).sum()
            num += y.shape[0]
        loss_val /= len(val_loader)
        acc_val = float(acc_val) / num
    return loss_val, acc_val


parser = argparse.ArgumentParser()
parser.add_argument("root_dir", help='directory contains data with each class in correspond directory')
parser.add_argument("model_type", help='Choose model to train (AlexNet, ResNet, VGG, EfficientNet)')
parser.add_argument("-e", "--epoch", help='Number of epochs', type=int, default=40)
parser.add_argument('-b', '--batch_size', help='batch data image to feed to model each iteration',type=int, default=64)
parser.add_argument('-lr', '--learning_rate', help='Initial Learning Rate', type=float, default=1e-2)


args = parser.parse_args()

root_dir = args.root_dir
epoch = args.epoch
batch_size = args.batch_size
model_type = args.model_type
lr = args.learning_rate

class_name = os.listdir(root_dir)
num_classes = len(class_name)
file_name = {k:os.listdir(os.path.join(root_dir, k)) for k in class_name}


# Check class in data
total = 0
for k,v in file_name.items():
    print(k, len(v))
    total += len(v)
    
print("Total:", total, 'images')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = T.Compose([
    T.Resize((256,256)),
    T.RandomCrop((224,224)),
    T.RandomHorizontalFlip(p=0.4),
    T.RandomRotation(20),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_file, val_file = split_train_val(file_name, ratio_train=0.8)

train_set = AnimalDataset(root_dir, train_file, transform=transform_train)
val_set = AnimalDataset(root_dir, val_file, transform=transform_val)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

if model_type == 'alexnet':
    model = AlexNet(num_classes=num_classes)
elif model_type == 'vgg':
    model = VGG('VGG16', in_channels=3, num_classes=num_classes)
elif model_type == 'resnet':
    model = ResNet50(num_classes=num_classes)
elif model_type == 'efficientnet':
    model = EfficientNet('b0', num_classes=num_classes)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
train_loss, train_acc, val_loss, val_acc = train(model, optimizer, scheduler,train_loader, val_loader, model_type+'.pth',epoch,print_every=100)

plot_training_process(train_loss, train_acc, val_loss, val_acc)