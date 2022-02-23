import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def split_train_val(file_name, ratio_train, ratio_val=None):
    train_set = {}
    val_set = {}
    total = 0
    for k, v in file_name.items():
        train_sample = int(len(v)*ratio_train)
        val_sample = int(len(v)*ratio_val) if ratio_val is not None else len(v) - train_sample
        shuffle_index = np.random.permutation(len(v))
        train_index = shuffle_index[:train_sample]
        val_index = shuffle_index[train_sample:train_sample+val_sample]
        train_set[k] = [v[i] for i in train_index]
        val_set[k] = [v[i] for i in val_index]
    return train_set, val_set

def plot_images(imgs, num_col, num_row, normalize=False):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
    if normalize:
        imgs = imgs*std + mean
    num_img = imgs.shape[0]
    plt.figure()
    for i in range(1,num_img+1):
        
        plt.subplot(num_row, num_col, i)
        plt.imshow(imgs[i-1].permute(1,2,0))
        plt.axis('off')



def plot_training_process(train_loss, train_acc, val_loss, val_acc):
  plt.figure()
  plt.subplot(2,1,1)
  plt.plot(train_loss, 'b', label='Train')
  plt.plot(val_loss, 'r', label='Val')
  plt.legend()
  plt.xlabel('Iter')
  plt.ylabel('Loss')
  plt.figure()
  plt.subplot(2,1,2)
  plt.plot(train_acc, 'b', label='Train')
  plt.plot(val_acc, 'r', label='Val')
  plt.legend()
  plt.xlabel('Iter')
  plt.ylabel('Accuracy')
  plt.show()

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