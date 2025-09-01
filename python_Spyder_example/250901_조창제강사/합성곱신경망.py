# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:23:38 2025

@author: human
"""
"""
합성곱 연산: 커널값과 커널크기의 입력값을 
    곱한 후 더하는 연산
kernel_size: 풀링에서 이동하는 윈도우 크기를 의미
stride: 윈도우가 움직일 때 이동하는 간격
padding: 입력 데이터에 추가영역을 주는 기법
dilation : 커널 사이의 간격을 의미

Conv Bn Act Pool 순으로 레이어 쌓음
Max Pool : 윈도우 내 최대값을 추출하는 형태
    기본값 2x 2 matrix

"""
import tensorflow as tf
import pandas as pd 
((train_input, train_target),
     (test_input, test_target)) =tf.keras.datasets.fashion_mnist.load_data()
train_input.shape

train_input_scale = (train_input / 255)
test_input_scale = (test_input / 255)

inputs = tf.keras.layers.Input(shape=(28, 28,1))
x = tf.keras.layers.Conv2D(
    64, kernel_size = 3, 
    padding="same", strides=1)(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(
    128, kernel_size = 3, 
    padding="same", strides=1)(x)
x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.Dense(
    10, activation="softmax")(x)
model = tf.keras.Model(inputs, x)
model.summary()

model.compile(
    loss = "sparse_categorical_crossentropy", metrics=["acc"])
# 배치크기별로 전체 데이터를 5번 학습 진행해달라는 의미
hist = model.fit(
    train_input_scale, train_target, 
    epochs=5, batch_size=32)

hist = model.fit(
    train_input_scale, train_target, 
    epochs=10, batch_size=32)

# 4시 10분까지
inputs = tf.keras.layers.Input(shape=(28, 28,1))
x = tf.keras.layers.Resizing(224,224)(inputs)
x = tf.keras.layers.Conv2D(
    64, kernel_size = 3, 
    padding="same", strides=1)(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(
    128, kernel_size = 3, 
    padding="same", strides=1)(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(
    256, kernel_size = 3, 
    padding="same", strides=1)(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(
    512, kernel_size = 3, 
    padding="same", strides=1)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.Dense(
    10, activation="softmax")(x)
model = tf.keras.Model(inputs, x)
model.summary()

model.compile(
    loss = "sparse_categorical_crossentropy", metrics=["acc"])
hist = model.fit(
    train_input_scale, train_target, 
    epochs=10, batch_size=32)
# 25분 정리