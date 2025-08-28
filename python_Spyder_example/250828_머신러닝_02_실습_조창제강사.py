# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 13:56:10 2025

@author: human
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

%matplotlib auto 
df = sns.load_dataset("titanic")
# https://codingalzi.github.io/datapy/pandas_titanic.html

data = load_diabetes()
"""
bp: 평균 혈압
s1: 총 콜레스테롤
s2: LDL 
s3: HDL
s4: 총 콜레스테롤
s5: 혈청 수치
s6: 혈당 수치
"""
df = pd.DataFrame(data.data, columns=data.feature_names)
print(data.DESCR)
data.target # 당뇨병 진행정도 정량화


df = sns.load_dataset("mpg")
df
"""
mpg: 연비 (miles per gallon) → 타깃 변수, 숫자형, 단위: 마일/갤런

cylinders: 실린더 수 (예: 4, 6, 8)
displacement: 배기량 (cubic inches, 입방 인치) 
horsepower: 마력 (horsepower)
weight: 차량 무게 (lbs, 파운드)
acceleration: 0 → 60 mph 도달 시간 (초 단위)
model_year: 자동차 모델 연식 (예: 70 → 1970년)
origin: 제조국 코드, 1 → 미국 (USA), 2 → 유럽 (Europe), 3 → 일본 (Japan)
name: 차종 이름 (문자열, 예: ford pinto)
"""

"""
연비 예측(회귀)
"""

from sklearn.model_selection import train_test_split
df = sns.load_dataset("mpg")
df = df[~ df.isna().any(axis=1)].reset_index(drop=True)
df.shape
train, test = train_test_split(df, test_size=0.1, random_state=42)

# 샘플링이 제대로 됬는지 확인! 학습데이터와 검증 데이터 분포가 유사
plt.hist(train['mpg'], bins=10, density=True)
plt.hist(test["mpg"], bins=10, density=True, alpha = 0.5)
train.isna().sum()
test.isna().sum()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train[["displacement","horsepower", "weight"]], train["mpg"])
# R^2 0.7067 = 학습데이터의 총 변동 중에서 70.6% 설명 가능
model.score(train[["displacement","horsepower", "weight"]], train["mpg"].values)
pred = model.predict(test[["displacement","horsepower", "weight"]])

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(test["mpg"], pred) # 평균 3.4마일/갤런 정도의 오차가 발생하고 있다.
# 28분까지 성능 높여보기 :) baseline : mae: 3.4175
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

model = ElasticNet()
scale = StandardScaler()
scale.fit(train[["displacement","horsepower", "weight"]])
scaled_train = scale.transform(train[["displacement","horsepower", "weight"]])
scaled_test = scale.transform(test[["displacement","horsepower", "weight"]])
model.fit(scaled_train, train["mpg"])
model.score(scaled_train, train["mpg"])
pred = model.predict(scaled_test)
mean_absolute_error(test["mpg"], pred) # MAE: 3.4175 -> 3.3959

#################################################################
df = sns.load_dataset("mpg")
df = df[~ df.isna().any(axis=1)].reset_index(drop=True)
## 수정된부분
dummies = pd.get_dummies(df["cylinders"])+0
dummies.columns=[f"sil_{i}" for i in dummies.columns]

df = pd.concat([df, dummies], axis= 1)
df = pd.concat([df, pd.get_dummies(df["origin"])+0], axis= 1)
train, test = train_test_split(df, test_size=0.1, random_state=42)

df.columns
model = ElasticNet()
scale = StandardScaler()
scale.fit(train[["displacement","horsepower", "weight"]])
train[["displacement","horsepower", "weight"]] = scale.transform(
    train[["displacement","horsepower", "weight"]])
test[["displacement","horsepower", "weight"]] = scale.transform(
    test[["displacement","horsepower", "weight"]])

tr_x = train.drop(["mpg","acceleration", "name", "origin"],axis=1)
te_x = test.drop(["mpg","acceleration", "name", "origin"],axis=1)

model.fit(tr_x, train["mpg"])
model.score(tr_x, train["mpg"])

pred = model.predict(te_x)
mean_absolute_error(test["mpg"], pred) # MAE: 3.3959 -> 3.213
mean_absolute_error(test["mpg"], pred) # 모델 연도를 고려했을 때 MAE: 3.3959 -> 2.54

