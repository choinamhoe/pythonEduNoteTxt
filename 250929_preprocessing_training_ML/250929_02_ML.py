import tensorflow as tf
import glob, cv2, os, tqdm
import numpy as np 
import pandas as pd

os.chdir("E:/choinamhoe/lacture_github_250926after/250929/청경채")

#### 15분 시작
tr_files = glob.glob("train/**/label.csv")

dfs = []
for file in tr_files:
    df = pd.read_csv(file)
    dfs.append(df)
# train data에 대한 모든 케이스에 대해 img_name과 잎의 무게
df = pd.concat(dfs, ignore_index = True)

filepath = df["img_name"][0]
# img_path
# meta data 경로 변수로 만들기(meta_path) # 27분 
# train/CASE_뭐시기/image 혹은 meta/파일명
f"train/{filepath.split('_')[0]}/image/{filepath}"
f"train/{filepath.split('_')[0]}/meta/{filepath.split('.')[0]}.csv"
#
[f"train/{i.split('_')[0]}/image/{i}" for i in df["img_name"]]
[f"train/{i.split('_')[0]}/meta/{i.split('.')[0]}.csv" for i in df["img_name"]]

img_paths = []
meta_paths = []
for i in df["img_name"]:
    img_path=f"train/{i.split('_')[0]}/image/{i}"
    meta_path = f"train/{i.split('_')[0]}/meta/{i.split('.')[0]}.csv"
    img_paths.append(img_path)
    meta_paths.append(meta_path)
df["img_path"] = img_paths
df["meta_path"] = meta_paths

### 35분 시작
meta_data = pd.read_csv(df["meta_path"][0])
"""
내부온도 관측치 평균, 내부온도 관측치의 최소값
내부습도 관측치 평균, 최소값
CO2관측치 평균
EC관측치 총합
총추정광량 총합
"""
numeric_df = meta_data.select_dtypes(include = 'number')
pd.DataFrame({
    "내부온도관측치_mean":[numeric_df["내부온도관측치"].mean()],
    "내부온도관측치_min":[numeric_df["내부온도관측치"].min()],
    })

agg_dict = {
    "내부온도관측치":["mean","min"],
    "내부습도관측치":["mean","min"],
    "CO2관측치":[np.mean],
    "EC관측치":"sum",
    "총추정광량":"sum"
    }
agg_df = numeric_df.agg(agg_dict)
agg_df_flat = agg_df.T.stack().to_frame().T

col_names = []
for i, j in agg_df_flat.columns:
    col_names.append(f"{i}_{j}")
agg_df_flat.columns = col_names

#### meta_fun 함수 형태로 구현 3시 3분 까지 진행

def meta_fun(path):
    # meta data 불러오기
    meta_data = pd.read_csv(path)
    # 하나의 행으로 집계
    agg_dict = {
        "내부온도관측치":["mean","min"],
        "내부습도관측치":["mean","min"],
        "CO2관측치":[np.mean],
        "EC관측치":"sum",
        "총추정광량":"sum"
        }
    agg_df = meta_data.agg(agg_dict)
    agg_df_flat = agg_df.T.stack().to_frame().T
    # 컬럼명 변경
    col_names = [] 
    for i, j in agg_df_flat.columns:
        col_names.append(f"{i}_{j}")
    agg_df_flat.columns = col_names
    return agg_df_flat

meta_fun(df["meta_path"][0])

meta_dfs = []
for meta_path in tqdm.tqdm(df["meta_path"]):
    res = meta_fun(meta_path)
    meta_dfs.append(res)
meta_df=pd.concat(meta_dfs)
df.shape
### 12분 시작
# 이미지 경로 읽어와서 픽셀 개수 새는 함수 만들어볼 예정
import matplotlib.pyplot as plt 
%matplotlib auto

img_path = df["img_path"][0]
img = cv2.imread(img_path)
# H 색상(0~180), S 채도(0~255), V 명도(0~255)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 청록색 추출
lower = np.array([30, 0, 0], dtype=np.uint8)
upper = np.array([100, 255, 200], dtype=np.uint8)
mask2 = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(img, img, mask = mask2)
# RGB 값 모두 0 인 부분을 흰색으로 변경
result[(result[...,0]==0)&(result[...,1]==0)&(result[...,2]==0)]=255
#plt.imshow(result)
# 청경채의 면적을 계산하기 위해서 사용
cv2.countNonZero(mask2) # 픽셀 값이 0이 아니면 1로 계산해서 합산
plt.imshow(mask2)

### 함수로 만들어보겠습니다. 24
def img_fun(path):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 0, 0], dtype=np.uint8)
    upper = np.array([100, 255, 200], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask2)

#type1
img_dfs = []
for img_path in tqdm.tqdm(df["img_path"]):
    result = img_fun(img_path)
    img_dfs.append(result)

#type 2
[img_fun(i) for i in tqdm.tqdm(df["img_path"])]

# type 3
import joblib
resluts = joblib.Parallel(n_jobs=-5)(
    joblib.delayed(img_fun)(i) for i in tqdm.tqdm(df["img_path"]))


## 33분 시작
df["true_pixel"] = resluts
train_df = pd.concat([df, meta_df.reset_index(drop=True)],axis=1)
train_df.columns
train_df["img_path"]
te_df = pd.DataFrame({"img_path":glob.glob("test/image/*")})

meta_paths = []
for img_path in te_df["img_path"]:
    meta_path =img_path.replace(
        "image","meta").replace(
            "png","csv").replace("jpg","csv")
    meta_paths.append(meta_path)
te_df["meta_path"] = meta_paths

### 4시까지 te_df에 meta_data활용한 변수와 img를 활용한 면적 변수 만들기
# type1
meta_dfs = []
for i in tqdm.tqdm(te_df["meta_path"]):
    meta_dfs.append(meta_fun(i))
# type 2
[meta_fun(i) for i in tqdm.tqdm(te_df["meta_path"])]

# type 3
meta_dfs = joblib.Parallel(n_jobs=-2)(
    joblib.delayed(meta_fun)(i) for i in tqdm.tqdm(te_df["meta_path"]))

te_meta_df = pd.concat(meta_dfs,ignore_index=True)

# 이미지 변수
result = joblib.Parallel(n_jobs=-2)(
    joblib.delayed(img_fun)(i) for i in tqdm.tqdm(te_df["img_path"]))

te_df["true_pixel"] = result
test_df = pd.concat([te_df, te_meta_df],axis = 1)

train_df
test_df

### 15분까지 모델 학습 진행
import lightgbm as lgb
model = lgb.LGBMRegressor(random_state = 42)
train_df_dropna = train_df.dropna()
tr_x=train_df_dropna.drop(
    ["img_name","leaf_weight","meta_path","img_path"],axis=1)
tr_y=train_df_dropna["leaf_weight"]
te_x = test_df.drop(["img_path","meta_path"],axis=1)

model.fit(tr_x, tr_y)
pred = model.predict(te_x)

submission = pd.read_csv("sample_submission.csv")
submission["leaf_weight"] = pred
submission.to_csv("01_submission.csv",index=False)

dl_model = tf.keras.models.load_model("model/test.h5")

tf.keras.Model(dl_model.input, dl_model.layers[-2].output).summary()
tf.keras.Model(
    dl_model.input, dl_model.get_layer("dense_4").output).summary()
dl_model.summary()

feature_extract_model = tf.keras.Model(
    dl_model.input, dl_model.layers[-2].output)

img_path = tr_df["img_name"][0]
def preprocessing(img_path):
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
    feature = feature_extract_model.predict(new_x[np.newaxis])
    return feature
##### preprocessing 함수 활용해서 tr_df 와 te_df 에 변수 추가하기 
##### 35분

import lightgbm as lgb
model = lgb.LGBMRegressor(random_state = 42)
features = [preprocessing(i) for i in tqdm.tqdm(train_df["img_path"])]

#################### 다음시간 계속
# 기존 train데이터+ 딥러닝으로 추출한 변수
tr_concat = pd.concat([
    train_df, 
    pd.DataFrame(np.concatenate(features))
    ],axis=1)
train_df_dropna = tr_concat.dropna() # 결측데이터 제거
tr_x=train_df_dropna.drop(
    ["img_name","leaf_weight","meta_path","img_path"],axis=1)
tr_y= train_df_dropna["leaf_weight"]

# 기존 test데이터 + 딥러닝으로 추출한 변수
te_features = [preprocessing(i) for i in tqdm.tqdm(test_df["img_path"])]

te_concat = pd.concat([
    test_df,
    pd.DataFrame(np.concatenate(te_features))
    ],axis= 1)

te_x = te_concat.drop(["img_path","meta_path"],axis=1)
tr_x.shape, te_x.shape
model.fit(tr_x, tr_y)
pred = model.predict(te_x)

submission = pd.read_csv("sample_submission.csv")
submission["leaf_weight"] = pred
submission.to_csv("02_submission.csv",index=False)