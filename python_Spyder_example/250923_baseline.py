# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:07:10 2025

@author: human
"""

import os
import pandas as pd
os.chdir("E:/최남회/파이썬개발에대한파일모음/235801_2021 농산물 가격예측 AI 경진대회")

# 1. 간단한 모델 만들어서 제출하기

# 2. 원본 데이터에서 Train데이터 만들기

# 3. 파생변수 만들어보기


"""
1. 간단한 모델 만들어서 제출하기
 1) 시각화
"""
train = pd.read_csv("public_data/train.csv")
# 시계열 데이터가 맞는지 확인
# 시계열 - 시간에 따라 주기성이 뚜렷하고 패턴이 있는 데이터

train["date"] = pd.to_datetime(train["date"])

temp = train[train["date"].dt.year == 2016]
import matplotlib.pyplot as plt
%matplotlib auto
plt.plot(temp.iloc[:,2])
temp = temp.iloc[:,2]
temp = temp[temp>0]

plt.plot(temp)
#### 10분
#2016, 17, 18, 19 같이 그려서 패턴이 있는지 확인해보겠습니다.
# 연도 2016, 배추 거래량(2번째 컬럼) 추출
train.columns
temp = train.loc[
    train["date"].dt.year == 2016, '배추_거래량(kg)']
temp = temp[temp>0].reset_index(drop=True) # 0~365로 시각화되게
plt.plot(temp.index, temp)

temp = train.loc[
    train["date"].dt.year == 2017, '배추_거래량(kg)']
temp = temp[temp>0].reset_index(drop=True)# 0~365로 시각화되게
plt.plot(temp.index, temp)

temp = train.loc[
    train["date"].dt.year == 2018, '배추_거래량(kg)']
temp = temp[temp>0].reset_index(drop=True)# 0~365로 시각화되게
plt.plot(temp.index, temp)

### 18분까지 for 문으로 변경
years = list(range(2016,2019))
n = 6

selected_column = train.columns[n]
print(selected_column)
for year in years:
    temp = train.loc[
        train["date"].dt.year == year, selected_column]
    temp = temp[temp>0].reset_index(drop=True)
    plt.plot(temp.index, temp)
    
### 25분
def vis_fun(df, years, col):
    print(col)
    for year in years:
        temp = df.loc[df["date"].dt.year == year, col]
        temp = temp[temp>0].reset_index(drop=True)
        plt.plot(temp.index, temp)

n=6
selected_column = train.columns[n]
selected_column =  '마늘_거래량(kg)'
vis_fun(train, list(range(2016,2019)), selected_column )    

train.columns

pum = "마늘"
train[f"{pum}_가격(원/kg)"]
train[f"{pum}_거래량(kg)"]

pd.DataFrame({
    "date":train["date"], 
    "origin":train[f"{pum}_가격(원/kg)"],
    "shift_1":train[f"{pum}_가격(원/kg)"].shift(1),
    "shift_2":train[f"{pum}_가격(원/kg)"].shift(2),
    "shift_3":train[f"{pum}_가격(원/kg)"].shift(3),
    "origin2":train[f"{pum}_가격(원/kg)"],
    "shift_m1":train[f"{pum}_가격(원/kg)"].shift(-1),
    "shift_m2":train[f"{pum}_가격(원/kg)"].shift(-2),
    })

pum = "마늘"
selected_column = f"{pum}_가격(원/kg)"
temp = train.copy()

df_list = list()
for lag in list(range(28, -1, -1)):
    # 28일 전부터 현재 시간까지 selected_column 데이터 추출
    df_list.append(temp[selected_column].shift(lag))
x = pd.concat(df_list,axis=1) 
x.columns = [f"lag_{i:02d}" for i in range(28,-1,-1)] # 컬럼명 변경

temp = train.copy()
df_list = list()
for lag in [1, 2, 4]:
    lag = -lag*7
    df_list.append(temp[selected_column].shift(lag))
y = pd.concat(df_list,axis=1) 
y.columns = [f"pred_{i:02d}" for i in [1, 2, 4]]

tr_df = pd.concat([x, y], axis = 1)

##### 20분 시작
tr_df = tr_df.dropna()

from lightgbm import LGBMRegressor

model_w1 = LGBMRegressor(random_state=42)
model_w1.fit(tr_df.iloc[:,:-3], tr_df.iloc[:, -3])

model_w2 = LGBMRegressor(random_state=42)
model_w2.fit(tr_df.iloc[:,:-3], tr_df.iloc[:, -2])

model_w4 = LGBMRegressor(random_state=42)
model_w4.fit(tr_df.iloc[:,:-3], tr_df.iloc[:, -1])

#35분 시작