# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 08:58:25 2025

@author: human
"""
"""
데이터 다운로드 경로
http://ducj.iptime.org:5000/sharing/z9aBSwhGd
"""
import glob,cv2
import pandas as pd
file_dir = "E:/choinamhoe/images/250902/cat_and_dogs"
cats = glob.glob(f"{file_dir}/cat/*")
dogs = glob.glob(f"{file_dir}/dog/*")
len(cats), len(dogs), len(cats) + len(dogs)

cats_df = pd.DataFrame({"label":1, "path": cats})
dogs_df = pd.DataFrame({"label":0, "path": dogs})
df = pd.concat([cats_df, dogs_df],ignore_index=True)
#froc 는 비율로 샘플링 1 기재하면 전체 데이터 랜덤 샘플링
df = df.sample(frac=1, random_state = 42).reset_index(drop=True)
#n은 특정 개수 뽑을 때 사용 가능
#df = df.sample(n=100, random_state = 42).reset_index(drop=True)
df

"""
만들려고 하는 함수 : 파일 경로를 입력받아 이미지를 np.array로 변환하고
    변환된 array를 224, 224, 3 이미지 크기로 리사이즈후 
    반환하는 함수
"""
import matplotlib.pyplot as plt
%matplotlib auto
file = cats[0]
img = cv2.imread(file)
img = img[...,::-1] # 마지막 차원이 RGB 채널을 의미
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 244))
plt.imshow(img)

#40분 시작(함수 만들어보기)
def load_image(file):
    """
    경로 입력시 이미지로 불러와서 224,224,3 이미지로 변환
    """
    img = cv2.imread(file)
    img = img[...,::-1] # 마지막 차원이 RGB 채널을 의미
    img = cv2.resize(img, (224, 224))
    return img

import tensorflow as tf
#이름 변경 불가 사전에 정의된 약속(자바에서 인터페이스 같은 틀 만들기)
"""
자바에서 인터페이스와 동일한 부분.꼭 구현해야 하는 함수 정의
tf.keras.utils.Sequence라는 인터페이스를 상속받아 처리하는 방법으로
아래의 기능에 대한 메서드 설명
__init__ : 자바에서 생성자와 동일
__len__ : 1 에포크 안에 몇 배치가 있는지 확인하기 위해 들어가있는 함수
__getitem__: 제너레이터 호출했을 때 동작할 함수
    batch_size 에 맞는 x, y를 반환
on_epoch_end: 1에포크가 끝났을 때 동작할 함수 
"""
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size
    def __len__(self):
        
    def __getitem__(self, index):
        
    def on_epoch_end(self):
        
load_image(file)

#df, index, batch_size 활용해서 배치크기 만큼의 x,y 만들기
import numpy as np
df
index = 0
batch_size = 4

st = index*batch_size
en = (index+1)*batch_size

selected_df = df.iloc[st:en,:]
selected_df.shape
imgs = []
for path in selected_df['path']:
    img = load_image(path)
    imgs.append(img)
imgs = np.array(imgs)
imgs.shape
x = imgs
x
y = selected_df["label"].values
y

#df, index, batch_size 입력받으면 index에 대한 batch_size
#만큼의 x,y 변환
def get_item_fun(df, index, batch_size):
    st = index*batch_size
    en = (index+1)*batch_size

    selected_df = df.iloc[st:en,:]
    imgs = []
    for path in selected_df['path']:
        img = load_image(path)
        imgs.append(img)
    imgs = np.array(imgs)
    x = imgs
    y = selected_df["label"].values
    return x,y

### 17분
x,y = get_item_fun(df, 2, 4)
x.shape, y.shape

class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size,load_image, get_item_fun):
        self.df = df
        self.batch_size = batch_size
        self.load_image = load_image
        self.get_item_fun = get_item_fun
    def __len__(self):
        return int(np.ceil(self.df.shape[0]/self.batch_size))
    def __getitem__(self, index):
        x,y = self.get_item_fun(self.df,index,self.batch_size)
        return x,y
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1)
# 배치단위로 몇번 진행해야 전체 데이터를 훓을 수 있는지 나타낸 수
int(np.ceil(df.shape[0]/batch_size))
gen = MyGenerator(df, batch_size, load_image, get_item_fun)
x,y = gen.__getitem__(1)
#gen.df
#gen.batch_size
#gen.load_image(file)


###### 
#모델 학습 시작
import glob, cv2
import pandas as pd
import tensorflow as tf

def load_image(file):
    """
    경로 입력시 이미지로 불러와서 224,224,3 이미지로 변환
    """
    img = cv2.imread(file)
    img = img[...,::-1] # 마지막 차원이 RGB 채널을 의미
    img = cv2.resize(img, (224, 224))
    return img

def get_item_fun(df, index, batch_size):
    st = index*batch_size
    en = (index+1)*batch_size

    selected_df = df.iloc[st:en,:]
    imgs = []
    for path in selected_df['path']:
        img = load_image(path)
        imgs.append(img)
    imgs = np.array(imgs)
    x = imgs
    y = selected_df["label"].values
    return x,y


class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size,load_image, get_item_fun):
        self.df = df
        self.batch_size = batch_size
        self.load_image = load_image
        self.get_item_fun = get_item_fun
    def __len__(self):
        return int(np.ceil(self.df.shape[0]/self.batch_size))
    def __getitem__(self, index):
        x,y = self.get_item_fun(self.df,index,self.batch_size)
        return x,y
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1)
        
file_dir = "E:/choinamhoe/images/250902/cat_and_dogs"
cats = glob.glob(f"{file_dir}/cat/*")
dogs = glob.glob(f"{file_dir}/dog/*")
len(cats), len(dogs), len(cats) + len(dogs)

cats_df = pd.DataFrame({"label":1, "path": cats})
dogs_df = pd.DataFrame({"label":0, "path": dogs})
df = pd.concat([cats_df, dogs_df],ignore_index=True)
df = df.sample(frac=1, random_state = 42).reset_index(drop=True)

from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(
    df, random_state=42, 
    test_size=0.2, stratify=df["label"])
test_df, valid_df = train_test_split(
    valid_df, random_state=42, 
    test_size=0.5, stratify=valid_df["label"])
batch_size = 32
train_gen = MyGenerator(
    train_df, batch_size, load_image, get_item_fun)
valid_gen = MyGenerator(
    valid_df, batch_size, load_image, get_item_fun)

inputs = tf.keras.layers.Input((224,224,3))
x = tf.keras.layers.Conv2D(
    64, kernel_size=3, padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(
    128, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(
    256, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(
    512, kernel_size=3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

optimizer = tf.keras.optimizers.Adam(
    learning_rate = 0.001)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [tf.keras.metrics.BinaryAccuracy()]
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics
    )

history = model.fit(
    train_gen, validation_data = valid_gen
    ,epochs=10
    )

import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# 전이학습: 기존에 잘 만들어져 있는 모델을 가져와서, 그 모델을 활용하여 내 모델로 만드는 것을 의미
# 작은 데이터에서도 금방 성능을 높일 수 있어서 많이 활용

# 모바일 넷 에서 7, 7, 1280 의 변수만 뽑아서 변수를 활용해서 내 모델을 만듬
backbone = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3), include_top = False
)
backbone.trainable = False # 모델이 학습과정에서 weight가 갱신되지 않게 바뀜

inputs = tf.keras.layers.Input(shape=(224,224,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = backbone(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

optimizer = tf.keras.optimizers.Adam(
    learning_rate = 0.001)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [tf.keras.metrics.BinaryAccuracy()]
model.compile(
    optimizer=optimizer,
    loss = loss,
    metrics = metrics)

history = model.fit(
    train_gen, validation_data = valid_gen,
    epochs = 10)