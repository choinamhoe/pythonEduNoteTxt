## 사용자 정의 함수 및 클레스를 호출해서 사용하는 방법 
#sys.path.append로 패키지 및 모듈 가져오는 경로를 하나 더 추가하는 형태로 사용
import sys
sys.path.append("E:/choinamhoe/lacture_github_250926after/250930") ## utils 파일이 있는 경로
from utils import Distiller, MyGenerator
import tensorflow as tf 

### 다음 시간 이어서 진행
# 경량화 방식 - 지식증류, 양자화 
import os, glob, cv2
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
files = glob.glob("E:/choinamhoe/lacture_github_250926after/250930/cat_and_dogs/**/*")
df = pd.DataFrame({"path":files})
[os.path.dirname(i) for i in df["path"]]
df["label"] = [os.path.basename(os.path.dirname(i)) for i in df["path"]]

# 기존 코드 참조해서 25분까지 8: 1: 1로 데이터 분할 하고 제너레이터 만들어보기 
## 0. train_test_split
## 1. 데이터 읽어오는 함수
## 2. 제너레이터 tf.keras.utils.Sequence, __init__, __getitem__, __len__, on_epoch_end 작성하기

train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)
valid_df, test_df = train_test_split(valid_df, test_size = 0.5, random_state = 42)
train_df.shape, valid_df.shape, test_df.shape

batch_size = 16
tr_gen=MyGenerator(train_df,batch_size, is_train=True)
val_gen=MyGenerator(valid_df,batch_size, is_train=True)
x,y = next(iter(tr_gen))
x.shape,y.shape

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',    # 검증 손실 기준
    factor=0.5,            # 학습률 줄이는 비율 (50%)
    patience=6,            # 개선 없을 시 3 에폭 기다림
    verbose=1,
    min_lr=1e-7
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,            # 7 에폭 동안 개선 없으면 중단
    verbose=1,
    restore_best_weights=True
)

callbacks = [reduce_lr, early_stop]

inp = tf.keras.layers.Input((224, 224, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inp)
backbone = tf.keras.applications.EfficientNetB0(
    input_shape = (224, 224, 3), include_top = False)
x = backbone(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
out = tf.keras.layers.Dense(1)(x)
teacher = tf.keras.Model(inp, out)

teacher.compile(
    optimizer = "Adam",
    loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = ["acc"]
    )

## 코드는 다지분류라 가정
checkpoint_path = f'model/teacher.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False  
)
teacher.fit(
    tr_gen, validation_data = val_gen,
    epochs = 30,
    callbacks = [*callbacks, checkpoint],
    )
################################
# 15분 시작
#model = tf.keras.models.load_model("C:/Users/human/Downloads/teacher.h5")
#model.layers[-1].activation = tf.keras.activations.linear
#model.save("C:/Users/human/Downloads/teacher_model.h5")

teacher = tf.keras.models.load_model("E:/choinamhoe/lacture_github_250926after/250930/model/teacher_model.h5")
teacher.summary()

inp = tf.keras.layers.Input((224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
backbone = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3), include_top = False)
x = backbone(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
out = tf.keras.layers.Dense(1)(x) # Active function 제거!!
student = tf.keras.Model(inp, out)
student.summary()

distiller = Distiller(
    student, teacher, 
    tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )
distiller.compile(
    optimizer = tf.keras.optimizers.Adam(),
    metrics= tf.keras.metrics.BinaryCrossentropy()
    )
checkpoint_path = f'model/student.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False  
)

distiller.fit(
    tr_gen, validation_data = val_gen,
    epochs = 30,
    callbacks = [*callbacks, checkpoint],
    )
### 35 분
backbone.summary()
backbone.layers[1].kernel_size

