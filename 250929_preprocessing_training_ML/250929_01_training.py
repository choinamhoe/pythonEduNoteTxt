import tensorflow as tf
import glob, cv2, os, tqdm
import numpy as np 
import pandas as pd

os.chdir("E:/choinamhoe/lacture_github_250926after/250929/청경채")
tr_files = glob.glob("train/**/label.csv",recursive=True)

dfs = [pd.read_csv(file) for file in tr_files]
tr_df = pd.concat(dfs, ignore_index=True)


# type 1
tr_df["img_name"] = "export2/train/"+ tr_df["img_name"]
# type 2
tr_df["img_name"] = [f"export2/train/{i}" for i in tr_df["img_name"]]

cv2.imread(tr_df['img_name'][0])

### 35분 시작
row = tr_df.iloc[0].values
img_path = row[0]
img = cv2.imread(img_path)
H, W = img.shape[:2] # H 308 W 410
max_HW = np.max([H, W])
## 410 , 410, 3의 흰색 배경 생성
new_x = np.zeros((max_HW, max_HW, 3))
new_x[:] = 255

center = max_HW//2
half_H = np.min([H, W])//2 #원본 이미지의 높이의 절반
lower = center - half_H
upper = center + half_H
new_x[lower:upper] = img
plt.imshow(new_x.astype(int))

import matplotlib.pyplot as plt
%matplotlib auto
plt.imshow(img)
label = row[1]
########## 함수로 구현

def preprocessing(row, is_train=True):
    img_path = row[0]
    img = cv2.imread(img_path)
    H, W = img.shape[:2] # H 308 W 410
    max_HW = np.max([H, W])
    ## 410 , 410, 3의 흰색 배경 생성
    new_x = np.zeros((max_HW, max_HW, 3))
    new_x[:] = 255

    center = max_HW//2
    half_H = np.min([H, W])//2 #원본 이미지의 높이의 절반
    lower = center - half_H
    upper = center + half_H
    new_x[lower:upper] = img
    new_x = cv2.resize(new_x, (224,224))
    if is_train:
        y = row[1]
        return new_x, y
    else:
        return new_x

preprocessing(tr_df.iloc[0],is_train=False)
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, fun, is_train):
        self.data =df 
        self.batch_size = batch_size
        self.preprocessing = fun 
        self.is_train = is_train
    def __len__(self):
        return np.ceil(self.data.shape[0]/self.batch_size).astype(int) # 정수 반환
    def __getitem__(self, index):
        st = index * self.batch_size
        ed = (index + 1) * self.batch_size
        # batch 크기만큼 row 추출
        rows = self.data.values[st:ed]
        x_list = []
        y_list = []
        for row in rows:
            if self.is_train:
                # (224,224,3), (1,)
                x, y = self.preprocessing(row, self.is_train)
                y_list.append(y)
            else:
                # (224, 224, 3)
                x = self.preprocessing(row, self.is_train)
            x_list.append(x)
        # (batch_size, 224,224,3)
        bat_x = np.array(x_list)
        if self.is_train:
            # (batch_size, 1)
            bat_y = np.array(y_list)
            return bat_x, bat_y
        else:
            return bat_x
    def on_epoch_end(self):
        self.data = self.data.sample(frac = 1)

gen=MyGenerator(tr_df, 32, preprocessing, is_train=False)
x=next(iter(gen))
x.shape

gen=MyGenerator(tr_df, 32, preprocessing, is_train=True)
x,y=next(iter(gen))
x.shape,y.shape

from sklearn.model_selection import train_test_split
my_bin = [0, 20, 100, 200, 300, 1000]
tr_df["leaf_weight_bin"] = pd.cut(tr_df["leaf_weight"], bins = my_bin, include_lowest=True)
train_df, valid_df = train_test_split(
    tr_df, test_size=0.1, random_state=42, stratify=tr_df["leaf_weight_bin"])

#### 38 분까지 데이터 학습
# 어그멘테이션 주의점

tr_gen=MyGenerator(train_df, 32, preprocessing, is_train=True)
val_gen=MyGenerator(valid_df, 32, preprocessing, is_train=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="model/test.h5",
    monitor="val_loss",
    save_bast_only=True, save_weight_only=False)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",patience =6, factor=0.5 # 학습률 줄이는 비율
    )

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",patience =10,
    restore_best_weights = True
    )

callbacks = [checkpoint, reduce_lr, earlystop]
inp = tf.keras.layers.Input(shape=(224,224,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3), include_top = False)
backbone.trainable = True
x = backbone(x)
gap = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512)(gap)
x = tf.keras.layers.Dense(128)(x)
out = tf.keras.layers.Dense(1, activation="ReLU")(x)
model = tf.keras.Model(inp, out)
model.compile(
    optimizer = "Adam",
    loss = "MSE",
    metrics = "MAE"
    )
model.fit(
    tr_gen,
    validation_data = val_gen,
    epochs = 100, 
    callbacks = callbacks)

te_df = pd.read_csv("sample_submission.csv")
te_df["img_name"] = [f"export2/test/{i}" for i in te_df["img_name"]]

te_gen=MyGenerator(te_df, 32, preprocessing, is_train=False)
pred = model.predict(te_gen,verbose=1)
te_df["leaf_weight"] = pred

te_df.to_csv("00_submission.csv", index = False)
