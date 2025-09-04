# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 08:51:22 2025

@author: human
"""

"""
순환신경망 (RNN, Recurrent Neural Network): 순서가 있는 데이터 예측/분류할 때 높은 성능을 보이는 딥러닝 모델 중 하나

기울기 소실문제: 순환신경망은 Column 개수가 너무 많아지면 앞부분 컬럼에 대한 정보를 잃어버리는 문제가 발생
ex) 기온 예측하는데, 72시간 전 자료부터 현재까지 매우 길게 컬럼을 고려하면, 처음 입력된 모델이 72시간, 71시간 전 자료에 대한 정보는 거의 고려하지 않고 1시간, 2시간 전 자료만 고려하는 현상을 의미

LSTM/GRU: 기울기 소실 문제 때문에 RNN에서 LSTM과 GRU로 발전

어제 데이터는 정형적인 형태 
> 23시간 전데이터 부터 현재시간까지 24개 컬럼을 입력하고
    1시간뒤 2시간뒤 3시간뒤 3개의 컬럼을 예측
    (24, 1) -> (3, 1)

오늘 할 영화 리뷰 데이터 긍정/부정 분류
> 길이가 정해지지 않은 text를 입력해서 0,1 로 분류
    (??,1) -> (1,1)
"""

# 23분에 진행 
"""
https://wikidocs.net/book/2155 추천 내용

영화 리뷰 데이터: IMDB
리뷰 데이터 -> 긍정(1)/부정(0)
"""
import tensorflow as tf
(tr_x, tr_y), (te_x, te_y) = tf.keras.datasets.imdb.load_data(
    num_words=200)

print(tr_x.shape, tr_y.shape)
word_index = tf.keras.datasets.imdb.get_word_index()
# {단어: 숫자} -> {숫자:단어}
my_dict = dict()
range(len(word_index))
word_index.items() # [(key, value),...,(key, value)]
"""
items 는 딕셔너리 데이터를 [(key, value),...,(key, value)] 형태로 출력해주는 함수
"""
for key,value in word_index.items():
    my_dict.update({value+3:key})

my_dict.update({0:"<PAD>"})
my_dict.update({1:"<START>"})
my_dict.update({2:"<Unknown>"})
my_dict.update({3:"<UNUSED>"})

n = 0
texts =[]
for word in tr_x[n]:
    texts.append(my_dict[word])
" ".join(texts) # 0번째 리뷰의 내용
tr_y[n] # 긍정적인 리뷰(1)

# 단어의 길이가 동일하지 않은 것을 확인
word_lengths = []
for values in tr_x:
    word_lengths.append(len(values))

import matplotlib.pyplot as plt
%matplotlib auto
plt.hist(word_lengths,bins=30)
# 200~300 빈도가 많고 1000자까지 작성한분도 계심
import numpy as np
np.max(word_lengths) # 최고 길이 2494
 
# 길이가 모자르면 0으로 채우고 길이가 남으면 짤라서 정형화 시킴
tr_seq = tf.keras.preprocessing.sequence.pad_sequences(
    tr_x, maxlen=200)

from sklearn.model_selection import train_test_split

tr_seq, val_seq, tr_y, val_y = train_test_split(
    tr_seq, tr_y, test_size=0.2, random_state = 42)

tr_seq.shape, val_seq.shape, tr_y.shape, val_y.shape

# 5분에 시작
te_seq = tf.keras.preprocessing.sequence.pad_sequences(
    te_x, maxlen=200)

"""
embedding 첫번째 200이라 기재된 부분은 데이터 로드했던 아래부분의
tf.keras.datasets.imdb.load_data(num_words=200)
num_words 의 숫자라고 함
"""
inputs = tf.keras.layers.Input(shape=(200,))
x = tf.keras.layers.Embedding(200, 16)(inputs) # 16은 출력개수(변경 가능)
x = tf.keras.layers.SimpleRNN(8)(x) # 8은 출력 개수(변경 가능)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer = "adam", loss = "binary_crossentropy",
    metrics = ["accuracy"])
# keras 3점대 버전부터 사용하는 확장자 -> keras
# keras 1, 2점대 확장자 -> h5
# filepath = "file.{epoch:02d}-{val_loss:.2f}.h5"
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    "best-model.h5", save_best_only = True)
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    "model-{epoch:02d}-{val_loss:.2f}.h5")
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    patience = 3, restore_best_weights = True)

hist = model.fit(
    tr_seq, tr_y,
    epochs = 30, 
    batch_size = 32,
    validation_data = (val_seq, val_y),
    callbacks = [ckpt_cb, early_stop_cb]
    )

## 28분시작
new_model = tf.keras.Model(model.input,model.layers[1].output)
new_model.summary()
new_model.trainable = False

pred = new_model.predict(tr_seq[:2])
# NLU (자연어 이해 모델) : 컴퓨터가 텍스트를 숫자로 이해할 수 있게 임베딩하는 과정
pred.shape
# 위에서 진행했듯 실수형태로 임베딩을 할 수도 있음
# 아래는 정수 형태로 원핫 인코딩 통해서 임베딩
# 원핫인코딩
tr_oh = tf.keras.utils.to_categorical(tr_seq)
val_oh = tf.keras.utils.to_categorical(val_seq)

inputs = tf.keras.layers.Input(shape=(200,200))
x = tf.keras.layers.SimpleRNN(8)(inputs)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer = "adam", loss = "binary_crossentropy",
    metrics = ["accuracy"])
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    "model_v2-{epoch:02d}-{val_loss:.2f}.h5")
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    patience = 3, restore_best_weights = True)
hist = model.fit(
    tr_oh, tr_y,
    epochs = 30, 
    batch_size = 32,
    validation_data = (val_oh, val_y),
    callbacks = [ckpt_cb, early_stop_cb]
    )

###############################################


# 0은 <PAD> 1 <START> 2: <Unknown> 3: <UNUSED>
index_to_word = {index+3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"

" ".join([index_to_word[i] for i in tr_x[0]])
