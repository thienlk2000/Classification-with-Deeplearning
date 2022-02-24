# Classification with Deeplearning
This repo train a deeplearning model with arbitrary dataset
## 1.Dataset
We use a dataset for classification problem. Dataset has structure like this animal dataset on [Kaggle](https://www.kaggle.com/alessiocorrado99/animals10)

![file](https://github.com/thienlk2000/Classification-with-Deeplearning/blob/main/images/github1.JPG)

dataset directory contain folders corresponding to each class of dataset

## 2.Select model
Select a model to train from stratch. Here you can use some common deeplearning model for classification:
- AlexNet 
- VGG 
- GoogleNet 
- EfficientNet

## 3.Train
Train model on your dataset by specifying image folder, model type, epoch, batch size and learning rate
```bash
python train.py raw-img efficientnet --epoch 40 --batch_size 64 --learning_rate 1e-2
```

## 4.Detect
Detect image folder using a model that you have trained by specifying image folder, label file contain class name, model type and weight 
```bash
python detect.py new-data label.txt efficientnet model_b0.pth 
```

## 5.Result 
I train model restnet and efficientnet on this animal dataset. Resnet has accuracy 77% while EfficientNet has 85% on validation set
### Resnet
![file](https://github.com/thienlk2000/Classification-with-Deeplearning/blob/main/images/img_test_resnet.png)

### EfficientNet
![file](https://github.com/thienlk2000/Classification-with-Deeplearning/blob/main/images/loss_ef.JPG)
![file](https://github.com/thienlk2000/Classification-with-Deeplearning/blob/main/images/acc_ef.JPG)
![file](https://github.com/thienlk2000/Classification-with-Deeplearning/blob/main/images/img_test_efficient.png)

Because I train the model from scratch, it is easily to be overfit with data. EfficienNet is a little better than ResNet.
You can use data augmentation and fine-tuning a pre-trained network from ImageNet to have better result. 
