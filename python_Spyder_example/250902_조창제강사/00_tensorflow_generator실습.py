# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 09:10:12 2025

@author: human
"""

"""
데이터 다운로드 경로
http://ducj.iptime.org:5000/sharing/z9aBSwhGd

패키지 설치시 
pip install opencv-python
pip install pandas tensorflow keras
pip install scikit-learn
"""

import glob, cv2
import pandas as pd 
file_dir = "C:/Users/human/Downloads/cat_and_dogs"
cats = glob.glob(f"{file_dir}/cat/*")
dogs = glob.glob(f"{file_dir}/dog/*")
len(cats), len(dogs), len(cats) + len(dogs)

cats_df = pd.DataFrame({"label":1, "path":cats})
dogs_df = pd.DataFrame({"label":0, "path":dogs})
df = pd.concat([cats_df, dogs_df], ignore_index=True)
# frac 는 비율로 샘플링 1 기재하면 전체 데이터 랜덤 샘플링
df=df.sample(frac=1, random_state = 42).reset_index(drop=True)
# n 은 특정 개수 뽑을 때 사용 가능
# df.sample(n=100, random_state = 42).reset_index(drop=True)

# 함수 만들기 전에 코드 짜보기
"""
만들려고 하는 함수: 파일경로를 입력받아 이미지를 np.array로 변환하고
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

# 40분 시작(함수 만들어보기)
def load_image(file):
    """
    경로 입력시 이미지로 불러와서 224,224,3 이미지로 변환
    """
    img = cv2.imread(file)
    img = img[...,::-1] # 마지막 차원이 RGB 채널을 의미
    img = cv2.resize(img, (224, 244))
    return img

import tensorflow as tf
# 이름 변경 불가 사전에 정의된 약속
"""
__len__: 1에포크 안에 몇 배치가 있는지 확인하기 위해 들어가있는 함수
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

# df , index, batch_size 활용해서 배치크기만큼의 x, y 만들기
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
y = selected_df["label"].values

# df, index, batch_size 입력받으면 index에 대한 batch_size 만큼의 x,y 반환
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
    return x, y

x,y = get_item_fun(df, 2, 4)
x.shape, y.shape
### 17분
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, df, batch_size, 
            load_image, get_item_fun):
        self.df = df 
        self.batch_size = batch_size
        self.load_image = load_image
        self.get_item_fun = get_item_fun
gen = MyGenerator(df, batch_size, load_image, get_item_fun)
gen.df
gen.batch_size
gen.load_image(file)
# get_item_fun(df, index, batch_size)
# 27분

class MyGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, df, batch_size, 
            load_image, get_item_fun):
        self.df = df 
        self.batch_size = batch_size
        self.load_image = load_image
        self.get_item_fun = get_item_fun
    def __len__(self):
        return int(np.ceil(self.df.shape[0]/self.batch_size))
    def __getitem__(self, index):
        x, y = self.get_item_fun(
            self.df, index, self.batch_size)        
        return x, y
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1)
    
# 배치단위로 몇번 진행해야 전체 데이터를 훑을수 있는지 나타낸 수
int(np.ceil(df.shape[0]/batch_size))

gen = MyGenerator(df, batch_size, load_image, get_item_fun)
x,y = gen.__getitem__(1)        

for i,(x,y) in enumerate(gen):
    print(i)
## 39분 시작

##################
# 모델 학습 시작
import glob, cv2
import pandas as pd 

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
    return x, y

class MyGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, df, batch_size, 
            load_image, get_item_fun):
        self.df = df 
        self.batch_size = batch_size
        self.load_image = load_image
        self.get_item_fun = get_item_fun
    def __len__(self):
        return int(np.ceil(self.df.shape[0]/self.batch_size))
    def __getitem__(self, index):
        x, y = self.get_item_fun(
            self.df, index, self.batch_size)        
        return x, y
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1)
    
file_dir = "C:/Users/human/Downloads/cat_and_dogs"
cats = glob.glob(f"{file_dir}/cat/*")
dogs = glob.glob(f"{file_dir}/dog/*")
len(cats), len(dogs), len(cats) + len(dogs)

cats_df = pd.DataFrame({"label":1, "path":cats})
dogs_df = pd.DataFrame({"label":0, "path":dogs})
df = pd.concat([cats_df, dogs_df], ignore_index=True)
df=df.sample(frac=1, random_state = 42).reset_index(drop=True)

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


inputs = tf.keras.layers.Input(
    shape = (224,224,3))
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
    optimizer=optimizer,
    loss = loss,
    metrics = metrics)
model.summary()

history = model.fit(
    train_gen, validation_data = valid_gen,
    epochs = 10)

# 28분쯤 
