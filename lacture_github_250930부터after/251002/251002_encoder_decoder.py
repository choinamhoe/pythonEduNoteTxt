import os, tqdm, cv2, glob

import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib auto
from sklearn.model_selection import train_test_split
os.chdir(r"E:\choinamhoe\lacture_github_250926after\251002\00_mnist")
tr_files = glob.glob("train/*")
te_files = glob.glob("test/*")

tr_df = pd.DataFrame({"path":tr_files})
te_df = pd.DataFrame({"path":te_files})

train_df, val_df = train_test_split(
    tr_df, test_size = 0.2, random_state = 42)

train_df.shape, val_df.shape, te_df.shape

#### 이미지 읽어와서 
# x 는 28, 28, 3 으로 그대로 반환
# y는 56, 56, 3 으로 크기 늘려서 반환
# 데이터 범위는 0~1

file = train_df["path"].values[0]
file
img = cv2.imread(file)
img = img/255.
new_img = cv2.resize(img, (56,56))

def img_read(path):
    img = cv2.imread(path)
    img = img/255.
    new_img = cv2.resize(img, (56,56))
    return img, new_img

#앞으로 제너레이터는 유틸 파일로 빼서 불러와서 사용하는 걸로
#사용하겠습니다. ::)
import sys
sys.path.append(r"E:\choinamhoe\lacture_github_250926after")
import aiUtils

        
### 13분까지 제너레이터 getitem 부분 채워보기 
batch_size =32
tr_gen = aiUtils.MyGenerator(tr_df, batch_size, img_read)
x,y = next(iter(tr_gen))
x,y

## 40분까지 정리 및 valid_gen,test_gen 만들기
val_gen = aiUtils.MyGenerator(val_df, batch_size, img_read)
te_gen = aiUtils.MyGenerator(te_df, batch_size, img_read)
# 함수들을 불러오는 다양한 방법들
import tensorflow as tf
# case 1
tf.keras.layers

# case 2
from tensorflow.keras import layers
layers

# case 3
layers = tf.keras.layers
layers 

############################
# 인코더 부분 inp~ p2
# c3 잠재공간
# 디코더 부분 up2~c6
inp = layers.Input(shape = (28, 28, 3))
c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
p1 = layers.MaxPool2D()(c1)

c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
p2 = layers.MaxPool2D()(c2)

c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)

up2 = layers.UpSampling2D()(c3)
c4 = layers.Conv2D(64, 3, activation="relu", padding="same")(up2)

up3 = layers.UpSampling2D()(c4)
c5 = layers.Conv2D(32, 3, activation="relu", padding="same")(up3)

up4 = layers.UpSampling2D()(c5)
c6 = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(up4)

model = tf.keras.Model(inp,c6)
#7분
x,y = next(iter(tr_gen))
plt.imshow(y[0])

model.compile(optimizer = "adam", loss = "mse")
model.fit(
    tr_gen,
    epochs=5,
    validation_data = val_gen
    )
"""
Epoch 1/5
1875/1875 [==============================] - 342s 181ms/step - loss: 0.0996 - val_loss: 0.0990
Epoch 2/5
1875/1875 [==============================] - 334s 178ms/step - loss: 0.0993 - val_loss: 0.0990
Epoch 3/5
1875/1875 [==============================] - 343s 183ms/step - loss: 0.0993 - val_loss: 0.0990
Epoch 4/5
1875/1875 [==============================] - 339s 181ms/step - loss: 0.0993 - val_loss: 0.0990
"""
x,y = next(iter(te_gen))
y_pred = model.predict(x)
x.shape

n = 1
view_x = np.zeros((56,56,3))
view_x[:] = 255
view_x[14:-14,14:-14] = x[n]
view = np.concatenate([view_x, y[n], y_pred[n]], axis=1)
plt.imshow(view)
#x,y = next(iter(tr_gen))
#%matplotlib auto
#plt.imshow(y[0])

model = tf.keras.Model(inp, c6)
# 7분 
model.compile(optimizer = "adam", loss = "binary_crossentropy")
model.fit(
    tr_gen,
    epochs= 1,
    validation_data = val_gen
    )
"""
1875/1875 [==============================] - 361s 192ms/step - loss: 0.4860 - val_loss: 0.1053
"""
x,y = next(iter(te_gen))
y_pred = model.predict(x)
x.shape

n = 1
view_x = np.zeros((56,56,3))
view_x[:] = 255
view_x[14:-14,14:-14] = x[n]
view = np.concatenate([view_x, y[n], y_pred[n]], axis=1)
plt.imshow(view)
