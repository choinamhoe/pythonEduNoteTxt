# 20분 모델 학습하기 전단계까지 직접 짜보기 :)
# 유틸도 써보시고 앞에 코드 복기 + 정리한다는 느낌으로 하시면 좋을 거 같아요 ㅎ

import os, tqdm, cv2, glob, sys

import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib auto
from sklearn.model_selection import train_test_split

sys.path.append(r"E:\choinamhoe\lacture_github_250926after")
import aiUtils
# from aiUtils import MyGenerator
os.chdir(r"E:\choinamhoe\lacture_github_250926after\251002\00_mnist")

tr_files = glob.glob("train/*")
te_files = glob.glob("test/*")

df = pd.DataFrame({'path':tr_files})
test_df = pd.DataFrame({"path":te_files})

train_df, valid_df = train_test_split(df, test_size =0.2, random_state=42)

def fun(path):
    img = cv2.imread(path)
    img = img/255.
    y = cv2.resize(img, (56,56))
    return img, y

filepath = train_df["path"][0]
x,y = fun(filepath)
x.shape, y.shape

batch_size = 32
tr_gen = aiUtils.MyGenerator(train_df, batch_size, fun)
val_gen = aiUtils.MyGenerator(valid_df, batch_size, fun)
te_gen = aiUtils.MyGenerator(test_df, batch_size, fun)
x, y = next(iter(tr_gen))
x.shape, y.shape

## 30분 시작
layers = tf.keras.layers
# 인코더 부분
inp = layers.Input(shape=(28,28,3))
c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
p1 = layers.MaxPool2D()(c1)

c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
p2 = layers.MaxPool2D()(c2)

# 잠재 공간
c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)

# 디코더 부분
u2 = layers.UpSampling2D()(c3)
concat2 = layers.Concatenate()([u2, c2]) # 추가된 부분
# conv 연산하기 전에 인코더의 기존 정보를 추가로 넘겨 받음
c4 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat2)

u1 = layers.UpSampling2D()(c4)
concat1 = ltkwkayers.Concatenate()([u1, c1]) # 추가된 부분
c5 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat1)

u0 = layers.UpSampling2D()(c5)
out = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(u0)
model = tf.keras.Model(inp, out)
model.compile(optimizer='adam', loss = 'mse')

model.fit(
    tr_gen,
    epochs=5,
    validation_data = val_gen
    )
"""
Epoch 1/5
1500/1500 [==============================] - 340s 226ms/step - loss: 0.0014 - val_loss: 1.1925e-04
Epoch 2/5
1500/1500 [==============================] - 324s 216ms/step - loss: 9.5735e-05 - val_loss: 6.9382e-05
Epoch 3/5
1500/1500 [==============================] - 301s 201ms/step - loss: 6.2213e-05 - val_loss: 4.9000e-05
Epoch 4/5
1500/1500 [==============================] - 301s 200ms/step - loss: 4.2585e-05 - val_loss: 3.6224e-05
Epoch 5/5
1500/1500 [==============================] - 300s 200ms/step - loss: 3.1931e-05 - val_loss: 2.6029e-05
"""
%matplotlib auto

x,y = next(iter(te_gen))
y_pred = model.predict(x)
x.shape

n = 1
view_x = np.zeros((56,56,3))
view_x[:] = 255
view_x[14:-14,14:-14] = x[n]
view = np.concatenate([view_x, y[n], y_pred[n]], axis=1)

plt.imshow(view)

### loss 변경해서 다시 테스트
model.compile(optimizer='adam', loss = 'binary_crossentropy')
model.fit(
    tr_gen,
    epochs=5,
    validation_data = val_gen
    )
"""
Epoch 1/5
1500/1500 [==============================] - 342s 227ms/step - loss: 0.0982 - val_loss: 0.0981
Epoch 2/5
1500/1500 [==============================] - 342s 228ms/step - loss: 0.0981 - val_loss: 0.0980
Epoch 3/5
1500/1500 [==============================] - 335s 223ms/step - loss: 0.0980 - val_loss: 0.0980
Epoch 4/5
1500/1500 [==============================] - 320s 214ms/step - loss: 0.0980 - val_loss: 0.0980
Epoch 5/5
1500/1500 [==============================] - ETA: 0s - loss: 0.0980  
"""

y_pred = model.predict(x)

x.shape

n = 1
view_x = np.zeros((56,56,3))
view_x[:] = 255
view_x[14:-14,14:-14] = x[n]
view = np.concatenate([view_x, y[n], y_pred[n]], axis=1)

plt.imshow(view)
"""
학습이 진행 되지 않으면 
1. 모델의 구조
2. Loss function
3. 출력층의 활성화함수

3가지를 문제로 의심할 수 있음

오전에는 Encoder-Decoder 구조에서 출력층 sigmoid loss MAE로 학습했을 때
학습이 되지 않았고, binary cross entropy로 학습할 때는 학습이 진행됨

오후에 모델 구조를 Unet으로 변경하게 되면서 출력층 sigmoid loss MAE로 해도
학습이 진행되었음. 

해당 문제에서 가장 이상 적인 조합은 
출력층 활성화함수 sigmoid와 binary cross entropy지만 
입/출력 구조만 맞으면 어느정도 학습 가능하다는 것을 보여주는 사례에 해당
"""