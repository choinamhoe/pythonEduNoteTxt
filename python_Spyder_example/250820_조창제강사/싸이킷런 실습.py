# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:19:01 2025

@author: human
"""

from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
print(iris["DESCR"])
df = pd.DataFrame(
    data = iris.data, 
    columns = iris.feature_names)
df.loc[:, "target"] = [
    iris.target_names[i] 
    for i in iris.target]
df
# 이렇게 데이터를 쪼개는 목적은 AI 모델이 강건(Robust)한 성능을 보일 수 있게 하려고 하는 것
# 강건하다는 말은 데이터에 어느정도 노이즈가 있어도 틀리지 않고 잘 맞춘다는 의미
from sklearn.model_selection import train_test_split
# 층화추출은 비율을 맞춰서 뽑아주는 옵션
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["target"],
    shuffle=False)
train.shape, test.shape
train["target"].value_counts()
test["target"].value_counts()
# 셔플은 자료를 섞는 옵션
train, test = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=False)
#######################################################
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["target"],
    shuffle=True)

valid, test = train_test_split(
    test, test_size=0.5, random_state=42, 
    shuffle=True, stratify=test["target"])

train["target"].value_counts()
valid["target"].value_counts()
test["target"].value_counts()

## 정규화
# 특정 컬럼의 범위를 
# 0~1, -1~1, mu=0, sigma = 1 등의 형태로 변환하는 작업
max_value = train["sepal length (cm)"].max()
min_value = train["sepal length (cm)"].min()
scaled_values = (
    train["sepal length (cm)"] - min_value)/(max_value - min_value)
scaled_values.max(), scaled_values.min()

mu =  train["sepal length (cm)"].mean()
sigma = train["sepal length (cm)"].std()
standard_values = (train["sepal length (cm)"] - mu)/sigma
standard_values.mean(), standard_values.std()

from sklearn.preprocessing import StandardScaler, MinMaxScaler
standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

min_max_scaler.fit(train.drop("target",axis=1))
tr_x = min_max_scaler.transform(train.drop("target",axis=1))
val_x = min_max_scaler.transform(valid.drop("target",axis=1))


standard_scaler.fit(train.drop("target",axis=1))
tr_x = standard_scaler.transform(train.drop("target",axis=1))
val_x = standard_scaler.transform(valid.drop("target",axis=1))


tr_y = train["target"]
tr_y[tr_y=="setosa"] = '0'
tr_y[tr_y=="versicolor"] = '1'
tr_y[tr_y=="virginica"] = '2'


import numpy as np
tr_y = tr_y.astype(np.int32)
# 오늘은 개념보다는 코드 익숙해지는 것에 초점 :)
##########
min_max_scaler.transform(test.drop("target",axis=1))
standard_scaler.transform(test.drop("target",axis=1))



###########################
df = pd.DataFrame(
    data = iris.data, 
    columns = iris.feature_names)
df.loc[:, "target"] = [
    iris.target_names[i] 
    for i in iris.target]

df
# 원 핫 인코딩, 더미화
Y = df["sepal length (cm)"]
X = df.drop("sepal length (cm)",axis =1 )
dummies_var = pd.get_dummies(X['target']) + 0
X = pd.concat([X.drop("target", axis =1), dummies_var], axis = 1)
