# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:50:48 2025

@author: human
"""

import glob
import numpy as np 
import pandas as pd 
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl

"""
1. 전체 데이터 불러오기 
2. 시간 결측 없게 join 활용해서 만들기
3. df.interpolate(method="linear") 활용하여 보간하기
4. 2023, 2024년도를 기준으로 자료 MinMax Scaling 하기
5. 학습형식에 맞게 자료를 
    t-23, t-22... t-1, 현재, t+1, t+2, t+3 형태로 만들기
6. 2023, 2024년도 Train데이터와 validation 데이터로 만들기
7. 2025년도 Test 데이터로 사용하기
8. 데이터셋 / 데이터로더 만들기
9. 모델 구축하기
10. 모델 학습하기
11. 모델 예측하기
12. 예측값 원래의 기온 범위로 복원하기(re scaling)
13. MAE와 같은 지표로 성능평가하기
"""
files = glob.glob("E:/choinamhoe/csv/**/*csv",recursive=True)
# 전체 데이터 읽어와서 하나로 합치기 14분 시작
# 3개 파일 읽어와서 df라는 이름으로 합치기
#Hint : pd.concat([],axis=1)
dfs = []
for file in files:
    df = pd.read_csv(file, encoding="cp949")
    dfs.append(df)
df = pd.concat(dfs,ignore_index=True)
df

df.isna().sum()
df["일시"] = pd.to_datetime(df["일시"])
(df["일시"].diff()).value_counts()
# 2시간 없는 경우 1건 
# 1시간 없는 경우 1건

min_dt = df["일시"].min()
max_dt = df["일시"].max()
alltimes = pd.date_range(min_dt, max_dt, freq="h")
time_df = pd.DataFrame({"일시":alltimes})
min_dt
max_dt
alltimes
time_df

##pd.merge 활용해서 df와 time_df 결합하기 24분
tot_df = pd.merge(df, time_df, on="일시", how="outer")
tot_df
tot_df = pd.merge(df, time_df, on="일시", how="right")
tot_df
tot_df = tot_df.iloc[:,2:]
tot_df
tot_df.columns = ["times", "temp"]
tot_df
# 2시간 연속 결측 1건
# 1시간 결측 1건
(df["일시"].diff()).value_counts() 
tot_df[tot_df["temp"].isna()]

#35분 진행
np.where(tot_df["temp"].isna())
[8152, 8153]
checked_index = list(range(8149, 8158))
tot_df.loc[checked_index] # 보간되기전 값 확인
tot_df = tot_df.interpolate(method="linear")
tot_df.loc[checked_index] # 보간된 결과값 확인

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler
"""
2023, 2024 자료를 바탕으로 -1~1 범위로 
    변경되게 min값 max 값을 설정
"""
tr_index = tot_df["times"].dt.year.isin([2023,2024])
# tr_index = tot_df["times"].dt.year!=2025
scaler.fit(
    tot_df.loc[tr_index,["temp"]])
# 2023/2024년도 데이터 min max 스케일링 반영
tot_df.loc[tr_index,["temp"]] = scaler.transform(
    tot_df.loc[tr_index,["temp"]])

# 2025년도 데이터 min max 스케일링 반영
tot_df.loc[~tr_index,["temp"]] = scaler.transform(
    tot_df.loc[~tr_index,["temp"]])

"""
5. 학습형식에 맞게 자료를 
    t-23, t-22... t-1, 현재, t+1, t+2, t+3 형태로 만들기
3시 13분까지 진행
df.shift활용
pd.concat 활용
"""
tot_df
tot_df.shift(1)
tot_df.shift(2)
tot_df.shift(3)

pd.concat([
    tot_df,
    tot_df["temp"].shift(1),
    tot_df["temp"].shift(2),
    tot_df["temp"].shift(3),
    ],axis=1)

# shift 23 ~ shift 1
dfs = []
list(range(23, 0, -1))
for i in list(range(23, 0, -1)):
    _df = tot_df["temp"].shift(i)
    dfs.append(_df)
pd.concat(dfs,axis = 1)

# -23시간전, -22시간전, ..., 현재시간, 1시간후, 2시간후, 3시간후
dfs = []
list(range(23, -4, -1))
for i in list(range(23, -4, -1)):
    _df = tot_df["temp"].shift(i)
    dfs.append(_df)
train_form_df = pd.concat(dfs,axis = 1)
train_form_df.columns = [f"lag_{i}" for i in list(range(23, -4, -1))]

train_form_df = pd.concat([tot_df[["times"]], train_form_df], axis=1)
train_form_df

### 정리하고 26분 진행 6. 2023, 2024년도 Train데이터와 validation 데이터로 만들기

"""
# 33분까지
6. 2023, 2024년도 Train데이터와 validation 데이터로 만들기
7. 2025년도 Test 데이터로 사용하기
"""
tr_index = train_form_df["times"].dt.year.isin([2023, 2024])
tr_df = train_form_df[tr_index] # 2023, 2024년도 데이터 추출
te_df = train_form_df[~tr_index] # 2025년도 데이터 추출
te_df = te_df.reset_index(drop = True)

tr_df, val_df = train_test_split(
    tr_df, test_size=0.1, shuffle=True, random_state=42)




import matplotlib.pyplot as plt 
%matplotlib auto
df_2023 = train_form_df[train_form_df["times"].dt.year==2023].copy()
df_2024 = train_form_df[train_form_df["times"].dt.year==2024].copy()
df_2025 = train_form_df[train_form_df["times"].dt.year==2025].copy()

df_2023["times"].dt.date
plt.plot(range(df_2023.shape[0]), df_2023["lag_0"], alpha=0.3)
plt.plot(range(df_2024.shape[0]), df_2024["lag_0"], alpha=0.3)
plt.plot(range(df_2025.shape[0]), df_2025["lag_0"], alpha=0.3)

plot_df = train_form_df.copy()
plot_df["date"] = plot_df["times"].dt.date
plot_df = plot_df.drop(
    "times",axis=1).groupby("date").mean().reset_index()

plot_df["date"] = pd.to_datetime(plot_df["date"])
df_2023 = plot_df[plot_df["date"].dt.year == 2023]
df_2024 = plot_df[plot_df["date"].dt.year == 2024]
df_2025 = plot_df[plot_df["date"].dt.year == 2025]

plt.plot(range(df_2023.shape[0]), df_2023["lag_0"], alpha=0.3)
plt.plot(range(df_2024.shape[0]), df_2024["lag_0"], alpha=0.3)
plt.plot(range(df_2025.shape[0]), df_2025["lag_0"], alpha=0.3)
"""
8. 데이터셋 / 데이터로더 만들기
9. 모델 구축하기
10. 모델 학습하기
11. 모델 예측하기
12. 예측값 원래의 기온 범위로 복원하기(re scaling)
13. MAE와 같은 지표로 성능평가하기
"""
index = 0 
batch_size = 4
tr_df.iloc[:,1:25]
tr_df.iloc[:,25:]
x_index = list(range(1, 25))
y_index = list(range(25,28))
tr_df.iloc[:,x_index]
tr_df.iloc[:,y_index]
"""
0~ 4 index에 해당하는 데이터 추출하고
추출된 데이터 x에 해당하는 행, y에 해당하는 행

위에서 진행한 것 활용해서 getitem_fun 구현하기 4시 8분 시작.
getitem_fun은 data랑 index, x_index, y_index 입력하면
index 에 해당하는 x, y 값 나오는 함수
"""

def getitem_fun(df, index, x_index, y_index):
    x = df.iloc[index ,x_index].values
    y = df.iloc[index ,y_index].values
    return x, y

x,y = getitem_fun(tr_df, 0, x_index, y_index)
x.shape, y.shape

tr_df.shape[0]
len(tr_df)
class RNNDataset(torch.utils.data.Dataset):
    def __init__(self, df, x_index, y_index, getitem_fun):
        self.df = df 
        self.x_index = x_index
        self.y_index = y_index
        self.getitem_fun = getitem_fun
    def __len__(self):
        # 전체 데이터의 길이 
        return self.df.shape[0]
    def __getitem__(self, index):
        x, y = self.getitem_fun(
            self.df, index, self.x_index, self.y_index)
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x = torch.tensor(x, dtype = torch.float32)
        y = torch.tensor(y, dtype = torch.float32)
        return x, y

tr_dataset = RNNDataset(tr_df, x_index, y_index, getitem_fun)
val_dataset = RNNDataset(val_df, x_index, y_index, getitem_fun)
x,y= next(iter(tr_dataset))
x.shape, y.shape
tr_dataloader = torch.utils.data.DataLoader(
    tr_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=True)

x,y= next(iter(tr_dataloader))
x.shape, y.shape

## 24분 시작

from torch import nn 
import pytorch_lightning as pl
# input, output, 체널수(변수 개수)
#nn.RNN(24, 64, 1, batch_first  = True) 

"""
num_layers 고려한 변수 개수( 기온)
input_dim: 고려한 시차(23시차 전부터 현재 시차 까지 총 24개) 
hidden_dim: RNN 레이어에서 출력될 출력값의 개수
output_dim: 최종적으로 출력할 개수(1시간뒤 2시간뒤 3시간뒤 총 3개)
"""
# 아래는 기본틀
# class RNNModel(pl.LightningModule):
#     def __init__(
#             self, lr=0.001, 
#             input_dim = 24, hidden_dim = 64, 
#             num_layers = 1, output_dim = 3
#             ):
#         super().__init__()
        
#     def forward(self, x):
        
#     def training_step(self):
        
#     def validation_step(self):
        
#     def configure_optimizers(self):
    
####
#nn.RNN(24, 64, 1, batch_first = True) 
class RNNModel(pl.LightningModule):
    def __init__(
            self, lr=0.001, 
            input_dim = 24, hidden_dim = 64, 
            num_layers = 1, output_dim = 3
            ):
        super().__init__()
        self.lr = lr
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)        
    def forward(self, x):
        # model.predict 함수 구현 부분에 해당
        x, _ = self.rnn(x.squeeze())
        x = self.fc(x)
        return x
    def training_step(self, batch, batch_index):
        x, y = batch 
        y_hat = self(x.unsqueeze(-1))
        loss = nn.functional.mse_loss(y_hat , y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss 
    def validation_step(self, batch, batch_index):
        x, y = batch 
        y_hat = self(x.unsqueeze(-1))
        loss = nn.functional.mse_loss(y_hat , y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

x.shape
model = RNNModel()
model(x)
x,y= next(iter(tr_dataloader))
x.shape
trainer = pl.Trainer(default_root_dir="E:/choinamhoe/logs",max_epochs=20)
trainer.fit(model, tr_dataloader, val_dataloader)
