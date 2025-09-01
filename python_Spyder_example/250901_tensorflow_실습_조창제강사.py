# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 09:02:34 2025

@author: human
"""

"""
# pip install tensorflow==2 keras
# pip install tensorflow==2 keras 했을 때 뜨는 목록 중에서 가장 낮은 버전
# pip install tensorflow==2.16.1 keras
"""
import tensorflow as tf
import pandas as pd 
((train_input, train_target),
     (test_input, test_target)) =tf.keras.datasets.fashion_mnist.load_data()
import matplotlib.pyplot as plt 
%matplotlib auto
plt.imshow(train_input[0],cmap="gray_r")
train_input.shape
pd.Series(train_target).value_counts()
""" page 361
레이블 {
     0:"티셔츠", 1:"바지", 2:"스웨터", 3:"드레스", 
     4:"코트", 5:"샌달", 6:"셔츠, 7:"스니커즈", 8:"가방",9:"앵클 부츠"}
"""

train_input_scale = (train_input / 255)
test_input_scale = (test_input / 255)
28*28
train_input_scale.shape
train_input_scale = train_input_scale.reshape(-1, 28*28)
test_input_scale = test_input_scale.reshape(-1, 28*28)

train_input_scale.shape, train_target.shape

"""
입력층(input layer): 데이터를 입력 받는 층
은닉층(hidden layer): 입력층과 출력층 사이에 존재하는 층
출력층(output layer): 최종 출력을 결정하는 층

Multi class 문제: 출력 라벨이 동시에 발생할 수 없다 가정
Multi label 문제: 출력 라벨이 동시에 발생할 수 있다 가정

https://datainsider.tistory.com/entry/Multi-ClassMulti-Label-Multi-class%EC%99%80-multi-label%EC%9D%98-%EA%B0%9C%EB%85%90-%EB%B0%8F-%EC%B0%A8%EC%9D%B4
# 멀티 클레스 문제(출력층 activation = softmax)
# 멀티 라벨 문제(출력층 activation = sigmoid)
"""
# Sequential API
# 멀티 클레스 문제(출력층 activation = softmax)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
# 멀티 클레스에서는 softmax 멀티 라벨 문제에서는 sigmoid
model.add(tf.keras.layers.Dense(10, activation="softmax")) 
model.summary()
# loss function은 batch크기별로 최소화 하고 싶은 값
"""
sparce categorical crossentropy : 
    target 은 0~9까지 실수 값 1개
    출력층은 실수값 10개
categorical crossentropy :
    target은 실수값 10개
    출력층은 실수값 10개
"""
model.compile(
    loss = "sparse_categorical_crossentropy", metrics=["acc"])
# 배치크기별로 전체 데이터를 5번 학습 진행해달라는 의미
model.fit(train_input_scale, train_target, epochs=5, batch_size=32)
prob = model.predict(test_input_scale)
prob.shape
import numpy as np 
np.round(prob[:3,:],3)
pred = np.argmax(prob,axis=1)
test_target.shape
from sklearn.metrics import accuracy_score
accuracy_score(test_target,pred) # 정확도 계산

# 멀티 라벨 문제
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="sigmoid"))
model.summary()
model.compile(loss = "sparse_categorical_crossentropy", metrics=["acc"])
model.fit(train_input_scale, train_target, epochs=5, batch_size=32)
prob = model.predict(test_input_scale)
np.round(prob[:3,:],3)

"""
# type1
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
# type 2
model.add(tf.keras.layers.Dense(64, activation="relu"))
"""
# categorical crossentropy 예시
train_input_scale = (train_input / 255)
test_input_scale = (test_input / 255)
train_input_scale.shape
train_input_scale = train_input_scale.reshape(-1,28*28)
test_input_scale = test_input_scale.reshape(-1, 28*28)

train_input_scale.shape, train_target.shape
train_target_cat = pd.get_dummies(train_target)+0

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="sigmoid"))
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(
    loss = "categorical_crossentropy", metrics=["acc"], 
    optimizer = opt)
model.fit(
    train_input_scale, train_target_cat, epochs=50, batch_size=32)
prob = model.predict(test_input_scale)
np.round(prob[:3,:],3)

# 45분 시작
"""
optimizers: 모델 학습하는데 어떤 함수로 최적화할지와 학습하는 정도를 설정하는 함수
metrics: 모델 학습과 무관하게 batch_size 별로 성능평가 결과를 측정한 정도
loss function: 배치단위로 최소화 하고 싶은 함수
"""

"""
과적합되지 않았는지 확인 필요 
"""
"""
type1 : 학습데이터를 일정 비율로 
    학습데이터(80%)와 검증데이터(20%)로 다시 쪼개서 사용
"""
hist = model.fit(
    train_input_scale, train_target_cat,
    validation_split=0.2,
    epochs=50, batch_size=32)

"""
type2: train_test_split 활용해서 입력하는 방식
"""
from sklearn.model_selection import train_test_split
train_input_scale = (train_input / 255)
test_input_scale = (test_input / 255)
train_input_scale.shape
train_input_scale = train_input_scale.reshape(-1,28*28)
test_input_scale = test_input_scale.reshape(-1, 28*28)

train_target_cat = pd.get_dummies(train_target)+0

tr_x, val_x, tr_y, val_y = train_test_split(
    train_input_scale, 
    train_target_cat, test_size=0.2)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="sigmoid"))
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(
    loss = "categorical_crossentropy", metrics=["acc"], 
    optimizer = opt)

hist = model.fit(
    tr_x, tr_y,
    validation_data = (val_x, val_y),
    epochs=50, batch_size = 32
    )
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
################################################
"""
모델의 레이어는 어느정도는 쌓을수록 성능이 개선됨
모델의 유닛은 밑에 레이어로 갈 수록 점점 커지는 형태로 쌓는 것이 일반적

옵티마이저는 모델을 개선하는 함수인데 모델이 어느정도 학습되게 되면
미세하게 조절되야 예측성능이 올라가므로 학습률 조절이 필요
> 학습률 조절은 콜백함수를 통해 수행

Loss function 을 통해서도 예측성능에 영향을 받음.
튀는 값에 대해 학습에 영향을 덜받는 loss function(focal loss 등)들도 많이 개발됨
"""
from sklearn.model_selection import train_test_split

# 학습 및 검증 데이터 분할
((train_input, train_target),
     (test_input, test_target)
     ) =tf.keras.datasets.fashion_mnist.load_data()
# 스케일링 (-1~1 범위)
train_input_scale = (train_input / 127.5 -1)
test_input_scale = (test_input / 127.5 - 1)
train_input_scale.min(), train_input_scale.max()

train_input_scale = train_input_scale.reshape(-1,28*28)
test_input_scale = test_input_scale.reshape(-1, 28*28)

train_target_cat = pd.get_dummies(train_target)+0

tr_x, val_x, tr_y, val_y = train_test_split(
    train_input_scale, 
    train_target_cat, test_size=0.2)

inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
loss_fun = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = 0.05)
metric_fun = tf.keras.metrics.Accuracy()
callback_fun = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3
        )]
model.compile(
    loss = loss_fun, metrics=[metric_fun], 
    optimizer = opt)

hist = model.fit(
    tr_x, tr_y,
    callbacks = callback_fun,
    validation_data = (val_x, val_y),
    epochs=50, batch_size = 32
    )
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])

prob = model.predict(test_input_scale)
prob.shape
pred = np.argmax(prob,axis=1)
accuracy_score(test_target,pred)
# 45분 시작
#########################
# 과적합 방지
"""
드롭아웃 기법 활용 가능
조기종료 콜백 함수 활용 가능
learning  rate 스케줄러 활용 가능

드롭아웃: 특정 유닛을 랜덤하게 비활성화 해주는 기법
조기종료: 
    에포크 기준으로 n회 동안 validation loss 개선이 없으면 
    학습을 종료해주는 콜백함수
learning rate 스캐줄러:
    에포크 기준으로 n회 동안 validation loss 개선이 없으면 
    learning rate를 줄여주는 콜백 함수
    ReduceLROnPlateau 가 예시
"""
inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
loss_fun = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
metric_fun = tf.keras.metrics.CategoricalAccuracy()
callback_fun = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.05,
        patience=5
        )]
model.compile(
    loss = loss_fun, metrics=[metric_fun], 
    optimizer = opt)

hist = model.fit(
    tr_x, tr_y,
    callbacks = callback_fun,
    validation_data = (val_x, val_y),
    epochs=50, batch_size = 32
    )

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])

# 10분 진행
tf.keras.layers.BatchNormalization()


inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

########
## 40 분까지 모델 만들어보고 평가해보기
((train_input, train_target),
     (test_input, test_target)
     ) =tf.keras.datasets.mnist.load_data()

plt.imshow(train_input[0],cmap="gray_r")


