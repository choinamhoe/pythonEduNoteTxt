# -*- coding: utf-8 -*-
"""
pip install pillow torch torchvision pytorch_lightning
"""

import os, glob, cv2, tqdm
import pandas as pd
import numpy as np 
import torch
from torch import nn
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

os.chdir("E:/cjcho_work/250919/235697_제 2회 컴퓨터 비전 학습 경진대회_data")

files = glob.glob("dirty_mnist_2nd/*")
label_df = pd.read_csv("dirty_mnist_2nd_answer.csv")

labels = []
for i in label_df["index"]:
    labels.append(f"./dirty_mnist_2nd/{i:05d}.png")
label_df["img_path"] = labels    

# 데이터 분할 하는 코드 
train, valid = train_test_split(
    label_df, test_size=0.2, random_state=42)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

IMG_SIZE = 224
BATCH_SIZE = 32


file = files[0]
img = Image.open(file).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform(img)

img = cv2.imread(file)[...,::-1]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform(img)



def load_image(path, transform):
    img = cv2.imread(path)
    img = img[...,::-1] # BGR 2 RGB
    img = transform(img)
    return img 


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
            self, df, load_image_fun, transform,
            img_size = (224,224)):
        self.df = df
        self.img_size = img_size
        self.transform = transform
        self.load_image = load_image_fun
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img_path"]
        img = self.load_image(img_path, self.transform)
        label = np.array(self.df.iloc[idx,1:-1],np.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label
    
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

tr_dataset = ImageDataset(train, load_image, transform)
val_dataset = ImageDataset(valid, load_image, transform)

train_loader = torch.utils.data.DataLoader(
    tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

x,y = next(iter(train_loader))

x.shape, y.shape


class SimpleMobileNet(pl.LightningModule):
    def __init__(self, lr=2e-5, scheduler_patience = 3):
        super().__init__()
        self.lr = lr
        backbone  = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Identity()
        for p in backbone.features.parameters(): 
            p.requires_grad = False
        self.backbone = backbone
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 26),
            nn.Sigmoid()  
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        y = y
        loss = self.criterion(self(x), y)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

# 체크포인트 콜백: val_loss 기준으로 최적 모델 저장
checkpoint_cb = ModelCheckpoint(
    monitor='val_loss',      # 모니터링할 metric
    mode='min',              # val_loss가 낮을수록 좋음
    save_top_k=1,            # 가장 좋은 1개만 저장
    filename='best_model'    # 저장될 파일 이름
)

# 얼리스탑 콜백: val_loss 개선 없으면 조기 종료
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,              # 5 epoch 동안 개선 없으면 종료
    mode='min'
)

model = SimpleMobileNet()
trainer = Trainer(
    max_epochs=10,
    accelerator='auto', # GPU 자동 감지
    devices=1,# GPU 하나 사용 (없으면 CPU)
    callbacks=[checkpoint_cb, earlystop_cb],
)
trainer.fit(model, train_loader, val_loader)