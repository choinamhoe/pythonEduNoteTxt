import cv2 , tqdm, os, glob
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt 

%matplotlib auto

os.chdir("E:/cjcho_work/250926")
tr_files = glob.glob("train/*")
train = pd.DataFrame({"path":tr_files })

te_files = glob.glob("test/*")
test = pd.DataFrame({"path":te_files })

"""
index = 0
batch_size = 4
df = train.copy()

st = index*batch_size
ed = (index+1)*batch_size

paths = df["path"].values[st:ed]

x_list = []
y_list = []
for file_path in paths:
    x, y= preprocessing(file_path)
    # 어그멘테이션 필요시 추가
    x_list.append(x)
    y_list.append(y)
np.array(x_list).shape
np.array(y_list).shape
"""

def preprocessing(file_path):
    y = int(file_path.split("_")[-1].split(".")[0])
    x = cv2.imread(file_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    _y = np.zeros(10)
    _y[y] = 1
    y = _y
    return x, y

def batch_pocessing(df, index, batch_size):
    st = index*batch_size
    ed = (index+1)*batch_size
    paths = df["path"].values[st:ed]

    x_list = []
    y_list = []
    for file_path in paths:
        x, y= preprocessing(file_path)
        # 어그멘테이션 필요시 추가
        x_list.append(x)
        y_list.append(y)
    bat_x = np.array(x_list)
    bat_y = np.array(y_list)
    return bat_x, bat_y

#########################
# 40분 클레스 생성
df.shape[0]//4

class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, preprocessing, aug_fun=None):
        self.data = df
        self.batch_size = batch_size 
        self.preprocessing = preprocessing
        self.aug_fun = aug_fun
    def __len__(self):
        # 전체길이/batch수
        return self.data.shape[0]//self.batch_size

    def __getitem__(self, index):
        st = index * self.batch_size
        ed = (index + 1) * self.batch_size
        paths = self.data["path"].values[st:ed]
        
        x_list = []
        y_list = []
        for file_path in paths:
            x, y= self.preprocessing(file_path)
            if self.aug_fun:
                x = np.array([aug_fun(image=i)["image"] for i in x])
            x_list.append(x)
            y_list.append(y)
        bat_x = np.array(x_list)
        bat_y = np.array(y_list)
        return bat_x, bat_y
    
    def on_epoch_end(self):
        # 전체 데이터 셔플
        self.data = self.data.sample(frac = 1)

import albumentations as A
augs = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ]
aug_fun = A.Compose(augs)
batch_size = 4
tr_gen = MyGenerator(train, batch_size, preprocessing, aug_fun=None)
x, y = next(iter(tr_gen))
x.shape , y.shape

inputs = tf.keras.layers.Input(shape=(224,224,3))
x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
backbone = tf.keras.applications.MobileNetV3Small(
    input_shape=(224,224,3), include_top=False)
backbone.trainable = False
x = backbone(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

### 컴파일 및 fit 작성하기 2시 15분 시작
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(train, test_size = 0.2, random_state = 42)
batch_size = 32
tr_gen = MyGenerator(train_df.reset_index(drop=True), batch_size, preprocessing, aug_fun=None)
val_gen = MyGenerator(valid_df.reset_index(drop=True), batch_size, preprocessing, aug_fun=None)

model.compile(
    optimizer = "Adam",
    loss="categorical_crossentropy",
    metrics=["acc"]
    )

model.fit(
    tr_gen,
    validation_data= val_gen,
    epochs=30
    )

## 23분 시작