# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:23:10 2025

@author: human
"""

import torch, glob 
from torch import nn
import pandas as pd
import numpy as np 
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
import pytorch_lightning as pl

transform = transforms.Compose([
    transforms.Resize((224, 224)), # 이미지 크기 조절
    # tensor 타입으로 변환, 0~255 -> 0~1 범위 변경
    transforms.ToTensor(), 
    # 정규화 작업을 각각 채널별로 진행
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225])
    ])

def get_item_fun(df, index, transform):
    img_path = df.loc[index,"path"]
    img = Image.open(img_path).convert("RGB")
    x = transform(img)    
    label = df.loc[index, "label"]
    y = torch.tensor(label, dtype=torch.float32)
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
    
# 출력부분 삭제
#backbone = models.mobilenet_v2(
#    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1)
#backbone.classifier = nn.Identity()
# trainable = False
#for p in backbone.features.parameters():
#    p.requires_grad = False
    
class SimpleCNN(pl.LightningModule):
    def __init__(self, lr=0.005, patience = 3):
        super().__init__()
        self.lr = lr
        # 출력부분 삭제
        backbone = models.mobilenet_v2(
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Identity()
        # trainable = False
        for p in backbone.features.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid())
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
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
            "val_loss: ", loss, 
            prog_bar = True, on_epoch=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 80], gamma=0.1)
        return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }

file_dir = "C:/Users/human/Downloads/cat_and_dogs"
dogs = glob.glob(f"{file_dir}/Dog/*")
cats = glob.glob(f"{file_dir}/Cat/*")

dog_df = pd.DataFrame({"label":1, "path":dogs})
cat_df = pd.DataFrame({"label":0, "path":cats})

df = pd.concat([dog_df, cat_df]).sample(
    frac=1, random_state = 42).reset_index(drop=True)

# 8: 1: 1
tr_df, val_df = train_test_split(
    df, test_size = 0.2, stratify=df["label"],
    random_state = 42)
te_df, val_df = train_test_split(
    val_df, test_size = 0.5, stratify=val_df["label"],
    random_state = 42)

tr_dataset = BinaryImageDataset(
    tr_df.reset_index(drop=True), 
    transform, get_item_fun)
val_dataset = BinaryImageDataset(
    val_df.reset_index(drop=True), 
    transform, get_item_fun)

batch_size = 32
tr_loader = torch.utils.data.DataLoader(
    tr_dataset, batch_size =batch_size, 
    shuffle=True, num_workers=0)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size =batch_size, 
    shuffle=True, num_workers=0)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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

trainer = Trainer(
    default_root_dir="E:/choinamhoe/logs",
    max_epochs=10,
    accelerator='auto', # GPU 자동 감지
    devices=1,# GPU 하나 사용 (없으면 CPU)
    callbacks=[checkpoint_cb, earlystop_cb ],
)
pl_model = SimpleCNN()
trainer.fit(pl_model, tr_loader, val_loader)
