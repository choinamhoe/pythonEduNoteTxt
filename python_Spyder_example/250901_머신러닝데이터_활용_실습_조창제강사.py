# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:01:21 2025

@author: human
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import tensorflow as tf 
from sklearn.model_selection import train_test_split
wine = pd.read_csv("https://bit.ly/wine_csv_data")
wine["class"] = wine["class"].astype(np.int32)

train, test = train_test_split(
    wine, test_size=0.2, random_state=42)
valid, test = train_test_split(
    test, test_size=0.5, random_state=42)

train_x = train.drop("class", axis=1).values
train_y = train["class"].values
train_x.shape, train_y.shape

test_x = test.drop("class", axis=1).values
test_y = test["class"].values
test_x.shape, test_y.shape

valid_x = valid.drop("class", axis=1).values
valid_y = valid["class"].values
valid_x.shape, valid_y.shape

inputs = tf.keras.layers.Input(shape = (3,))
x = tf.keras.layers.Dense(64, activation = "relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation = "relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation = "relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation = "relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(
    1, activation = "sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

"""
train_x[:1].shape # 차원 확인
pred = model.predict(train_x[:1]) # 모델 돌리기전에 예측
loss = tf.keras.losses.BinaryCrossentropy()
loss(pred, train_y[:1])# 성능 평가 되는지 확인
"""
# 머신러닝 대비 추가된 사항
loss_fun = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
metric_fun = tf.keras.metrics.BinaryAccuracy()
model.compile(
    loss = loss_fun, metrics = [metric_fun], 
    optimizer=opt)
callback_fun = [
    tf.keras.callbacks.EarlyStopping(patience=15),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.05, patience=5)]

# 학습 및 검증데이터 입력은 그대로
# 에포크, 배치사이즈 라는 개념, 콜백함수라는 개념이 추가
# 하이퍼 파라미터 최적화까지 된 형태라 생각하면 됨
hist = model.fit(
    train_x, train_y,
    epochs = 100, batch_size = 64,
    callbacks = callback_fun,
    validation_data = (valid_x, valid_y))


import matplotlib.pyplot as plt 
#%matplotlib auto
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])

prob = model.predict(test_x)
pred = np.round(prob)

from sklearn.metrics import accuracy_score
accuracy_score(test_y, pred)

### 3시까지
df = sns.load_dataset("mpg")
df = df[~ df.isna().any(axis=1)].reset_index(drop=True)

tf.keras.layers.Dense(1, activation = "linear")
loss = tf.keras.losses.MeanAbsoluteError()

# 레이어 연산은 행렬 연산의 연속으로 이루어져있음
# 레이어별 beta1와 beta0
for layer in model.layers:
    if len(layer.weights):
        print(
            layer.weights[0].shape, # weight
            layer.weights[1].shape) # bias

"""
배치크기 단위로 학습될 때 마다 
위의 모델의 가중치를 갱신하는데 출력층부터 
순차적으로 갱신을 수행

갱신될 때 기존 가중치 대비 얼마나 갱신되는지에 대한 
가중치를 학습율이라 함
"""

df = sns.load_dataset("mpg")
df = df[~ df.isna().any(axis=1)].reset_index(drop=True)
train, test = train_test_split(
    df, test_size=0.2, random_state=42)
valid, test = train_test_split(
    test, test_size=0.5, random_state=42)

selected_columns = ["displacement", "horsepower",
                    "weight", "model_year"]

train_x = train[selected_columns].values
train_y = train["mpg"].values
train_x.shape, train_y.shape

test_x = test[selected_columns].values
test_y = test["mpg"].values
test_x.shape, test_y.shape

valid_x = valid[selected_columns].values
valid_y = valid["mpg"].values
valid_x.shape, valid_y.shape


inputs = tf.keras.layers.Input(shape = (4,))
x = tf.keras.layers.Dense(64, activation = "relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation = "relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation = "relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation = "relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(
    1, activation = "relu")(x) 
# relu 0~ 무한대
# linear -무한대~ 무한대
model = tf.keras.Model(inputs, outputs)
model.summary()
loss_fun = tf.keras.losses.MeanAbsoluteError()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
metric_fun = tf.keras.metrics.MeanAbsoluteError()
model.compile(
    loss = loss_fun, metrics = [metric_fun], 
    optimizer=opt)
callback_fun = [
    tf.keras.callbacks.EarlyStopping(patience=15),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.05, patience=5)]

hist = model.fit(
    train_x, train_y,
    epochs = 100, batch_size = 64,
    callbacks = callback_fun,
    validation_data = (valid_x, valid_y))

pred = model.predict(test_x)
pred.min()

import matplotlib.pyplot as plt 
#%matplotlib auto
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
