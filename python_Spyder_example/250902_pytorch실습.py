# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:01:49 2025

@author: human
"""

# pip install torch torchvision lightning
"""
텐서 플로우 대비 자유도가 높음
그래서 직접 작성해야 하는 부분이 상당수 많음
"""
import torch
from torch import nn
import glob, cv2
import pandas as pd
import numpy as np

file_dir = "E:/choinamhoe/images/250902/cat_and_dogs"
cats = glob.glob(f"{file_dir}/Cat/*")
dogs = glob.glob(f"{file_dir}/Dog/*")

len(cats), len(dogs), len(cats) + len(dogs)

cats_df = pd.DataFrame({"label":1, "path": cats})
dogs_df = pd.DataFrame({"label":0, "path": dogs})

df = pd.concat([cats_df, dogs_df],ignore_index=True)
#froc 는 비율로 샘플링 1 기재하면 전체 데이터 랜덤 샘플링
df = df.sample(frac=1, random_state = 42).reset_index(drop=True)
#n은 특정 개수 뽑을 때 사용 가능
#df = df.sample(n=100, random_state = 42).reset_index(drop=True)
df

"""
tensorflow 입력되는 이미지 순서가 B W H C
pytorch 입력되는 이미지 순서가 B C W H
B : batch, C: channel, W:width, H: height
"""

#pip install pillow
from PIL import Image
file = cats[0]
Image.open(file)
Image.open(file).convert("L") # L: 흑백, RGB: 컬러
img = Image.open(file).convert("RGB") # L: 흑백, RGB: 컬러

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224,224)), #이미지 크기 조절
    #tensor 타입으로 변환, 0~255 -> 0~1 범위 변경
    transforms.ToTensor(),
    #정규화 작업을 각각 채널별로 진행
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
    ])
transform(img)
transform(img).shape

class BinaryImageDataset(
        torch.utils.data.Dataset):
    def __init__(self):
        
    def __len__(self):
        
    def __getitem__(self, index):
        
index = 0
img_path = df.loc[index,"path"]
img = Image.open(img_path)
transform(img).shape, transform(img).dtype
x = transform(img)

label = df.loc[index, "label"]
y = torch.tensor(label, dtype=torch.float32)

# df, index 입력하면 x, y 반환하는 함수 만들기 55분 시작
def get_item_fun(df, index, transform):
    img_path = df.loc[index,"path"]
    img = Image.open(img_path)
    x = transform(img)    
    label = df.loc[index, "label"]
    y = torch.tensor(label, dtype=torch.float32)
    return x, y

get_item_fun(df, 0, transform)

class BinaryImageDataset(
        torch.utils.data.Dataset):
    def __init__(self,df,transform, get_item_fun):
        self.df = df
        self.transform = transform
        self.get_item_fun = get_item_fun
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        x, y = self.get_item_fun(
            self.df, index, self.transform)
        return x, y
    
class BinaryImageDataset(
        torch.utils.data.Dataset):
    def __init__(self, df, transform, fun):
        self.df = df
        self.transform = transform
        self.get_item_fun = fun
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        x, y = self.get_item_fun(
            self.df, index, self.transform)
        return x, y
    
my_dataset = BinaryImageDataset(
    df, transform, get_item_fun)
x,y = next(iter(my_dataset))    
x.shape,y.shape
#데이터로더 = 제너레이터

from sklearn.model_selection import train_test_split
tr_df, val_df = train_test_split(
    df, random_state=42, 
    test_size=0.2, stratify=df["label"])
te_df, val_df = train_test_split(
    val_df, random_state=42, 
    test_size=0.5, stratify=val_df["label"])       

tr_dataset = BinaryImageDataset(
    tr_df.reset_index(drop=True), transform, get_item_fun)
val_dataset = BinaryImageDataset(
    val_df.reset_index(drop=True), transform, get_item_fun)

batch_size = 32
tr_loader = torch.utils.data.DataLoader(
    tr_dataset, batch_size =batch_size, 
    shuffle=True, num_workers=0)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size =batch_size, 
    shuffle=True, num_workers=0)

my_dataset = BinaryImageDataset(
    df, transform, get_item_fun)
x,y = next(iter(my_dataset))    
x.shape,y.shape
# 데이터로더 = 제너레이터 

x, y = next(iter(tr_loader))
x.shape, y.shape

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(128*56*56,256)
        self.fc2 = nn.Linear(256,1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        print("conv1", x.shape)
        x = self.pool(x)
        print("pool", x.shape)
        x = self.conv2(x)
        print("conv2", x.shape)
        x = self.pool(x)
        print("pool", x.shape)
        x = self.flat(x)
        print("flat", x.shape)
        x = self.fc1(x)
        print("fc1", x.shape)
        x = self.fc2(x)
        print("fc2", x.shape)
        x = self.act(x)
        print("act", x.shape)
        return x

model = CNN()
model(x)
optim = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.BCEWithLogitsLoss() #criterion

import pytorch_lightning as pl
class SimpleCNN(pl.LightningModule):
    def __init__(self, model, lr=0.005, patience = 3):
        super().__init__()
        self.lr = lr 
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, x):
        x = self.model(x)
        return x
    def training_step(self, batch, batch_index):
        x, y = batch
        y = y.unsqueeze(1) 
        loss = self.criterion(self(x), y)
        return loss
    def validation_step(self, batch, batch_index):
        x, y = batch
        y = y.unsqueeze(1) 
        loss = self.criterion(self(x), y)
        self.log(
            "val loss: ", loss, 
            prog_bar = True, on_epoch=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

pl_model = SimpleCNN(model)
from pytorch_lightning import Trainer

trainer = Trainer(default_root_dir="E:/choinamhoe/logs",max_epochs=10,accelerator="auto")
trainer.fit(pl_model, tr_loader, val_loader)
## 4시 22분
## 모델 학습 직접 구현
model.eval() # 모델 학습되지 않게 수정
num_epochs = 10
device = torch.device("cpu") # gpu: torch.device("cuda:0")
for epoch in range(num_epochs):
    # train
    model.train() # 모델 가중치 업데이트 가능하게 변경
    train_loss = 0
    for x, y in tr_loader:
        x = x.to(device) # 장비 변경
        y = y.to(device) # 장비 변경
        optim.zero_grad() # 기울기 업데이트 되지 않게 설정
        y_hat = model(x) # 예측값 추론
        loss = criterion(y_hat, y.unsqueeze(1)) # loss 계산
        loss.backward()  # 모델 가중치 업데이트
        optim.step()
        train_loss += loss.item() * x.size(0)
        print(loss.item() * x.size(0))
    train_loss /= len(tr_loader.dataset)
    # valid
    model.eval()# 모델 학습되지 않게 수정 
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device) # 장비 변경
            y = y.to(device) # 장비 변경
            y_hat = model(x) # 예측값 추론
            # loss 계산
            loss = criterion(y_hat, y.unsqueeze(1)) 
            val_loss += loss.item() * x.size(0)            
            print(loss.item() * x.size(0))
        val_loss /= len(val_loader.dataset)


"""
교재 502 페이지도 유사 내용 존재 
"""

