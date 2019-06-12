# -*- coding: utf-8 -*-
#将数据按6：2：2分为训练集，验证集和测试集

import os
import numpy as np
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms
#设置固定的随机种子，保证训练数据的统一
np.random.seed(1001)

#以下是一些超参数和别的什么
DATA_DIR='fin_data/'
BATCH_SIZE=32

#将图片地址存入不同的列表，并设置标签
train_img=[]
valid_img=[]
test_img=[]
img_label=0
train_label=[]
valid_label=[]
test_label=[]
for i in os.listdir(DATA_DIR):
    path=DATA_DIR+i
    for j in os.listdir(path):
        img=path+'/'+j
        rua=np.random.randint(100)
        if rua<20:
            valid_img.append(img)
            valid_label.append(img_label)
        elif rua<40:
            test_img.append(img)
            test_label.append(img_label)
        else:
            train_img.append(img)
            train_label.append(img_label)
    img_label+=1

#制作dataset类以便dataloader读取
class MyDataset(Dataset):
    def __init__(self,img,label,transform=None):
        self.img=img
        self.label=label
        self.transform=transform
    def __getitem__(self, index):
        data=self.img[index]
        data=Image.open(data).convert('RGB')
        if self.transform is not None:
            data=self.transform(data)
        target=self.label[index]
        return data,target
    def __len__(self):
        return len(self.img)

#使用transform预处理数据，包括增强等

my_trans1=transforms.Compose([
        transforms.RandomResizedCrop(size=256,scale=(0.8,1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
])
my_trans2=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

#使用dataloader处理dataset
train_data=MyDataset(train_img,train_label,transform=my_trans1)
valid_data=MyDataset(valid_img,valid_label,transform=my_trans2)
test_data=MyDataset(test_img,test_label,transform=my_trans2)

train_loader=DataLoader(train_data,BATCH_SIZE,True)
valid_loader=DataLoader(valid_data,BATCH_SIZE,True)
test_loader=DataLoader(test_data,BATCH_SIZE,True)
