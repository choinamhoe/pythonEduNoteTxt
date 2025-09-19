
"""
conda create -n tf_28 python==3.10 
conda activate tf_28 
pip install numpy==1.23 tensorflow==2.8.0  spyder protobuf==3.20 opencv-python scikit-learn 
pip install pandas
pip install tqdm

# 데이터 다운로드 링크 
http://ducj.iptime.org:5000/sharing/zzoIWTM3D
"""

import os, glob
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
os.chdir("E:/최남회/파이썬개발에대한파일모음/235697_제 2회 컴퓨터 비전 학습 경진대회_data")

files = glob.glob("dirty_mnist_2nd/*")
label_df = pd.read_csv("dirty_mnist_2nd_answer.csv")
#label_df["img_path"] = files

f"{1000:05d}"
labels = []
for i in label_df["index"]:
    labels.append(f"./dirty_mnist_2nd/{i:05d}.png")
label_df["img_path"] = labels    

# 데이터 분할 하는 코드 
train, test = train_test_split(
    label_df, test_size=0.2, random_state=42)
valid, test = train_test_split(
    test , test_size=0.5, random_state=42)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

import cv2,tqdm
file = train.loc[0,"img_path"]
img = cv2.imread(file)
img.shape # 256 256 3

# type1( 이미지 크기 확인)
img_sizes = [cv2.imread(i).shape for i in tqdm.tqdm(files)]

# type2
img_sizes = []
for i in tqdm.tqdm(files):
    img = cv2.imread(i)
    img_sizes.append(img.shape)

pd.Series(img_sizes).value_counts()

#파일 경로 입력하면 numpy 로 반환하는 함수
def load_image(path):
    img = cv2.imread(path)
    # type1
    img = img[...,::-1] # BGR 2 RGB
    img = cv2.resize(img , (224,224))
    # type 2
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 
img = load_image(file)
"""
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, df):
        
    def __len__(self):
        # 배치의 총 길이 출력 되게 
        # 배치가 100이고 자료 수가 40000이면 40000/100인 
        # 400이 나올 수 있게
    def __getitem__(self, index):
        # 인덱스가 입력되면 배치 크기만큼의 x, y 가 반환되게
        # batch_size가 4이면 
        # (4, 256, 256, 3), (4, 26) 26은 라벨 갯수
    def on_epoch_end(self):
        # 매 에포크가 끝났을 때 동작할 행동 기재
"""

index = 1
batch_size = 4
# 배치 크기 만큼 자료 추출하게 작업
start_index = index * batch_size
end_index = (index+1) * batch_size
start_index, end_index
extract_df = train.iloc[start_index:end_index,:]
# 33분 진행

x = []
for i in extract_df.iloc[:,-1]:
    img = load_image(i)
    x.append(img)
x = np.array(x)
x.shape
y = extract_df.iloc[:,1:-1].values

# 48
# index 기재하면 배치 크기만큼 x, y 나오는 함수
def fun(index, df, batch_size, load_image):
    start_index = index * batch_size
    end_index = (index+1) * batch_size
    extract_df = df.iloc[start_index:end_index,:]
    # 86~ 90번째줄과 같은 형태
    x= np.array(
        [load_image(i) for i in extract_df.iloc[:,-1]])
    y = extract_df.iloc[:,1:-1].values
    return x, y

x,y = fun(0, train, 4, load_image)
x.shape, y.shape
# load_image, fun
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, df, img_load_fun, extract_fun):
        self.batch_size = batch_size
        self.df = df
        self.img_load_fun = img_load_fun
        self.extract_fun = extract_fun
    def __len__(self):
        # df.shape[0] 은 행 갯수
        # 반올림(행 개수/배치크기)  정수형으로 반환
        # int(self.df.shape[0]%self.batch_size) 로 기재해도 무방해보임
        return int(np.ceil(self.df.shape[0]/ self.batch_size))

    def __getitem__(self, index):
        x, y = self.extract_fun(
            index, self.df, 
            self.batch_size, self.img_load_fun)
        return x, y
    def on_epoch_end(self):
        # 매 에포크가 끝났을 때 동작할 행동 기재
        self.df = self.df.sample(frac=1)
gen = MyGenerator(32, train, load_image, fun)
x,y = next(iter(gen))
train.shape
gen.__len__() # 1250 출력 
x.shape, y.shape #  ((32, 224, 224, 3), (32, 26))

# 35분 진행 
4%3
train.sample(frac=1)

batch_size = 32
tr_gen = MyGenerator(batch_size, train, load_image, fun)
val_gen = MyGenerator(batch_size, valid, load_image, fun)
te_gen = MyGenerator(batch_size, test, load_image, fun)

for x,y in tr_gen:
    print(x,y)
    break

inputs = tf.keras.layers.Input(shape=(224,224,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
#model = tf.keras.Model(inputs, x)
#model.summary()
#tf.keras.applications.efficientnet.preprocess_input()
backbone = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False)
#backbone = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=True)

x = backbone(x)
x = tf.keras.layers.GlobalAvgPool2D()(x)
x = tf.keras.layers.Dense(512)(x)
outputs = tf.keras.layers.Dense(26,  activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [
    EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights = True),
    ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor= 0.1, patience=3)
    ]

model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = ["acc"],
    )

model.fit(
    tr_gen, 
    validation_data = val_gen,
    epochs = 10,
    callbacks = callbacks
    )

pred_df = pd.read_csv("./sample_submission.csv")

for i,idx in enumerate(tqdm.tqdm(pred_df["index"])):
    file = f"./test_dirty_mnist_2nd/{idx:05d}.png"
    img = load_image(file)
    img = cv2.resize(img, (224,224))
    img = img[np.newaxis]
    pred = model.predict(img)
    pred = (pred>0.5).astype(int)
    pred_df.iloc[i, 1:] = pred

pred_df.to_csv("./first_submit.csv",index=False)
