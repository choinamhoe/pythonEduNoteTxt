# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:04:53 2025

@author: human
"""

import glob
import pandas as pd
### 3개 파일
files = glob.glob("E:/choinamhoe/csv/**/*csv",recursive=True)
df1 = pd.read_csv(files[0],encoding="cp949")
df2 = pd.read_csv(files[1],encoding="cp949")
df3 = pd.read_csv(files[2],encoding="cp949")

df = pd.concat([df1,df2,df3],ignore_index=True)

dfs = []
for file in files:
    df = pd.read_csv(file,encoding="cp949")
    dfs.append(df)
df = pd.concat(dfs,ignore_index=True)

df = pd.concat([pd.read_csv(i,encoding="cp949") for i in files],
    ignore_index=True)
df = df.iloc[:,2:]
df
df.columns = ["times","temp"]
df["times"] = pd.to_datetime(df["times"])
min_dt = df["times"].min()
max_dt = df["times"].max()
#시작 시간부터 끝 시간까지 특정 간격으로 시간 데이터 모두 만들어달라
all_dt = pd.date_range(min_dt,max_dt, freq="h")
time_df = pd.DataFrame({"times":all_dt})

final_df= pd.merge(df, time_df, on="times", how="outer")
final_df
final_df.isna().sum() #결측치 수
na_idx = final_df["temp"].isna()
final_df[final_df["temp"].isna()] # 결측자료 확인

final_df["temp"] = final_df["temp"].interpolate(method="linear")
final_df["temp"].interpolate(method="linear")[na_idx]
final_df["temp"].bfill()[na_idx]
final_df["temp"].ffill()[na_idx]

final_df["temp"] = final_df["temp"].interpolate(method="linear")
final_df["temp"].interpolate(method="linear")[na_idx]
final_df["temp"].bfill()[na_idx]
final_df["temp"].ffill()[na_idx]

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(-1, 1))
train_idx = final_df["times"].dt.year!=2025
# -1~1 사이로 스케일링 학습
scale.fit(final_df.loc[train_idx, ["temp"]]) 

final_df.loc[train_idx, ["temp"]] = scale.transform(
    final_df.loc[train_idx, ["temp"]])
final_df.loc[~train_idx, ["temp"]] = scale.transform(
    final_df.loc[~train_idx, ["temp"]])

# 32분 시작
sample = final_df[:3].copy()
sample.shift(1)
pd.concat([sample, sample["temp"].shift(1), sample["temp"].shift(2)], axis=1)

concat_df= pd.concat([
    final_df["temp"].shift(23),
    final_df["temp"].shift(22),
    final_df["temp"].shift(21),
    final_df["temp"].shift(20),
    final_df["temp"].shift(19),
    final_df["temp"].shift(18),
    final_df["temp"].shift(17),
    final_df["temp"].shift(16),
    final_df["temp"].shift(15),
    final_df["temp"].shift(14),
    final_df["temp"].shift(13),
    final_df["temp"].shift(12),
    final_df["temp"].shift(11),
    final_df["temp"].shift(10),
    final_df["temp"].shift(9),
    final_df["temp"].shift(8),
    final_df["temp"].shift(7),
    final_df["temp"].shift(6),
    final_df["temp"].shift(5),
    final_df["temp"].shift(4),
    final_df["temp"].shift(3),
    final_df["temp"].shift(2),
    final_df["temp"].shift(1),
    final_df["temp"],
    final_df["temp"].shift(-1),
    final_df["temp"].shift(-2),
    final_df["temp"].shift(-3),
],axis=1)

concat_df.columns = [f"lag_{i}" for i in list(range(23,-4, -1))]

# 40분 시작
dfs = []
for i in list(range(23,-4, -1)):
    _df = final_df[["temp"]].shift(i)
    _df.columns = [f"lag_{i}"]
    dfs.append(_df)
concat_df = pd.concat(dfs,axis=1)
concat_df["times"] = final_df["times"]
concat_df = concat_df.dropna()
concat_df = concat_df.reset_index(drop=True)
concat_df.iloc[:,:24]
concat_df.iloc[:,24:-1]


x_index = range(24)
y_index = range(24, 27)
x = concat_df.iloc[:,x_index]
y = concat_df.iloc[:,y_index]

from sklearn.model_selection import train_test_split

tr_df = concat_df[concat_df["times"].dt.year!=2025]
tr_df, val_df = train_test_split(
    tr_df , test_size=0.2, random_state=42)
te_df = concat_df[concat_df["times"].dt.year==2025]
# 8:1:1
#te_df, val_df = train_test_split(
#    val_df, test_size=0.5, random_state=42)
tr_df.shape, val_df.shape, te_df.shape
### 12분 시작

index = 0
batch_size = 4
st = index*batch_size
ed = (index+1)*batch_size
x = tr_df.iloc[st:ed,x_index].values
y = tr_df.iloc[st:ed,y_index].values

# 24 분까지 구현
def getitem_fun(df, index, batch_size, x_index, y_index):
    st = index*batch_size
    ed = (index+1)*batch_size
    x = df.iloc[st:ed,x_index].values
    y = df.iloc[st:ed,y_index].values
    return x, y

x, y = getitem_fun(tr_df, 0, 32, x_index, y_index)
x.shape, y.shape

df.shape[0]/batch_size

import tensorflow as tf
import numpy as np 
class RNNGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, df, batch_size, x_index, y_index, getitem_fun):
        self.df = df 
        self.batch_size = batch_size
        self.x_index = x_index
        self.y_index = y_index
        self.getitem_function = getitem_fun
    def __len__(self):
        # 배치크기를 기준으로 1에포크 도는데 필요한 반복수
        # 배치크기가 4면 4를 기준으로 전체 데이터수를 나눈 값 +1
        return int(np.ceil(df.shape[0]/self.batch_size))
    def __getitem__(self, index):
        x, y =self.getitem_function(
            self.df, index,  self.batch_size, 
            self.x_index, self.y_index)
        return x, y
    def on_epoch_end(self):
        self.df.sample(frac=1)

tr_gen = RNNGenerator(
    tr_df,  32, x_index, y_index, 
    getitem_fun=getitem_fun)
val_gen = RNNGenerator(
    val_df,  32, x_index, y_index, 
    getitem_fun=getitem_fun)

x,y = next(iter(tr_gen))

for x,y in tr_gen:
    break

x.shape,y.shape

inputs = tf.keras.layers.Input(shape = (24, 1))
x = tf.keras.layers.SimpleRNN(64, activation="tanh")(inputs)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(3)(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer  = optimizer,
    loss = loss,
    metrics= ["MAE"]
    )

epochs = 10
hist = model.fit(
    tr_gen, 
    validation_data =val_gen,
    epochs= epochs,
    )

pred = model.predict(te_df.iloc[:,:24].values)
y_true = scale.inverse_transform(te_df.iloc[:,24:-1].values)
y_pred = scale.inverse_transform(pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)

# 25분 진행
####################################
inputs = tf.keras.layers.Input(shape = (24, 1))
x = tf.keras.layers.LSTM(64, return_sequences = True)(inputs)
x = tf.keras.layers.LSTM(64, return_sequences = False)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(3)(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer  = optimizer,
    loss = loss,
    metrics= ["MAE"]
    )

epochs = 10
hist = model.fit(
    tr_gen, 
    validation_data =val_gen,
    epochs= epochs,
    )

pred = model.predict(te_df.iloc[:,:24].values)
y_true = scale.inverse_transform(te_df.iloc[:,24:-1].values)
y_pred = scale.inverse_transform(pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)

# 35분 시작
###############################################
# 교재 544~630 page
inputs = tf.keras.layers.Input(shape = (24, 1))
x = tf.keras.layers.GRU(64, return_sequences = True)(inputs)
x = tf.keras.layers.GRU(64, return_sequences = False)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(3)(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.MeanSquaredError()

model.compile(
    optimizer  = optimizer,
    loss = loss,
    metrics= ["MAE"]
    )

epochs = 10
hist = model.fit(
    tr_gen, 
    validation_data =val_gen,
    epochs= epochs,
    )

pred = model.predict(te_df.iloc[:,:24].values)
y_true = scale.inverse_transform(te_df.iloc[:,24:-1].values)
y_pred = scale.inverse_transform(pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)
