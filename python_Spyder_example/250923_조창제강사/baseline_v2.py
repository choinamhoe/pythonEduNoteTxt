# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 11:38:02 2025

@author: human
"""

import os
import pandas as pd
os.chdir("E:/cjcho_work/250923/235801_2021 농산물 가격예측 AI 경진대회")

train = pd.read_csv("public_data/train.csv")
train["date"] = pd.to_datetime(train["date"])

########### 함수로 짜기 위해서 임시로 만든 코드
column = train.columns[3]
df = train.copy()
temp = df[column]
lagtime = 28

x = list()
lag_list = list(range(lagtime, -1, -1))
for lag in lag_list:
    x.append(temp.shift(lag))
x = pd.concat(x, axis=1)
x.columns = [f"lag_{i:02d}" for i in lag_list]
x

y = list()
lag_list = [-7, -14, -28]
for lag in lag_list:
    y.append(temp.shift(lag))
y = pd.concat(y, axis=1)
y.columns = [f"pred_{i:02d}" for i in lag_list]

result = pd.concat([x, y], axis=1)
###########
def preprocessing(column, df, lagtime):
    temp = df[column]
    x = list()
    lag_list = list(range(lagtime, -1, -1))
    for lag in lag_list:
        x.append(temp.shift(lag))
    x = pd.concat(x, axis=1)
    x.columns = [f"lag_{i:02d}" for i in lag_list]
    
    y = list()
    lag_list = [-7, -14, -28]
    for lag in lag_list:
        y.append(temp.shift(lag))
    y = pd.concat(y, axis=1)
    y.columns = [f"pred_{i:02d}" for i in lag_list]

    result = pd.concat([x, y], axis=1)
    return result
    
# 한 컬럼에 대해서 모델 1주일 뒤 2주일뒤 4주일 뒤 학습하고 models에 저장하기
column = train.columns[3]
tr_df = preprocessing(column, train, 28)
tr_df = tr_df.dropna()
tr_x = tr_df.iloc[:,:-3]

models = {}
from lightgbm import LGBMRegressor
for target in ["pred_-7","pred_-14","pred_-28"]:
    tr_y = tr_df[target]
    model = LGBMRegressor(random_state=42)
    model.fit(tr_x, tr_y)
    models.update({f"{column}_{target}":model})
    
    
# 모든 컬럼에 대해서 모델 1주일 뒤 2주일뒤 4주일 뒤 학습하고 models에 저장하기
models = {}
for column in train.columns[2:]:
    tr_df = preprocessing(column, train, 28)
    tr_df = tr_df.dropna()
    tr_x = tr_df.iloc[:,:-3]
    
    from lightgbm import LGBMRegressor
    for target in ["pred_-7","pred_-14","pred_-28"]:
        tr_y = tr_df[target]
        model = LGBMRegressor(random_state=42)
        model.fit(tr_x, tr_y)
        models.update({f"{column}_{target}":model})

models
# 12시 15분 시작
submission = pd.read_csv("sample_submission.csv")
for idx, date in enumerate(submission["예측대상일자"]):
    pred_date, pred_week = date.split("+")
    # pred_date 2020-09-29
    pred_week = int(pred_week.replace("week", "")) #pred_week 1 or 2 or 4
    test = pd.read_csv(f"public_data/test_files/test_{pred_date}.csv")
    alldata = pd.concat([train,test,pd.DataFrame(
        {"date":[pd.to_datetime(pred_date)]})]) 
    alldata["date"]=pd.to_datetime(alldata["date"])
    alldata = alldata.sort_values("date") # 날짜 순 정렬
    for column in submission.columns[1:]:
        # 예측 데이터 
        te_data = preprocessing(column, alldata, 28).fillna(0)
        te_x = te_data.iloc[[-1],:-3] # 입력값
        f"{column}_pred_{pred_week*-7}" #  '배추_가격(원/kg)_pred_-7'
        select_model = models[f"{column}_pred_{pred_week*-7}"] # 선택된 모델
        pred = select_model.predict(te_x)
        submission.loc[idx, column] = pred[0]

submission.to_csv("submit.csv",index=False)
