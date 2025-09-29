"""
데이터 출처
https://dacon.io/competitions/official/235897/data

http://ducj.iptime.org:5000/sharing/D25wQpZbu
"""
import os, glob, tqdm, cv2
import numpy as np 
import pandas as pd

os.chdir("E:/cjcho_work/250929/청경채")
files = glob.glob("sample/**/label.csv", recursive=True)
file = files[0]
## files 파일들 다 합쳐서 하나의 
## dataframe 만들어보는 시간 44분
pd.read_csv(file)

# type 1
dfs = [pd.read_csv(file) for file in files]

# type 2
dfs=list()
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

#### 55분 까지 이미지 경로 수정하기
df['img_name']

# CASE01/image/CASE01_01.png 형태로 바로 읽어올 수 있게
# type 1
img_paths = []
for file in df['img_name']:
    base_dir = file.split("_")[0]
    filepath = f"sample/{base_dir}/image/{file}"
    img_paths.append(filepath)
df["img_name"] = img_paths

# type 2
df["img_name"] = [f"sample/{i.split('_')[0]}/image/{i}" 
                  for i in df["img_name"]]

file = df["img_name"][0]
img = cv2.imread(file)

import matplotlib.pyplot as plt
%matplotlib auto

plt.imshow(img[...,::-1])

# H는 색상, S는 채도, V는 명도
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# H 의 범위는 0 ~ 180, S 0~255, V 0~255
lower = np.array([0, 0, 200], dtype=np.uint8)
upper = np.array([180, 255, 255], dtype=np.uint8)

mask_background = cv2.inRange(hsv, lower, upper)
plt.imshow(mask_background)
mask_background.shape

# 청록색 뽑기 위해서 30~ 100의 색상, 채도 명도는 모두다 선택되게 
lower = np.array([30, 0, 0], dtype=np.uint8)
upper = np.array([100, 255, 255], dtype=np.uint8)
mask_target = cv2.inRange(hsv, lower, upper)
plt.imshow(mask_target)

### 25분 시작
###########
mask_forground = cv2.bitwise_not(mask_background)
plt.imshow(mask_target)
mask = cv2.bitwise_and(mask_target, mask_forground )
plt.imshow(mask)
# 마스크 처리
result = cv2.bitwise_and(img, img, mask = mask)
# 검은색 부분 흰색으로 변경
result[(result[...,0]==0)&(result[...,1]==0)&(result[...,2]==0)] =255
plt.imshow(result)

### for문으로 파일 저장하기
os.makedirs("export", exist_ok= True)
for file in df["img_name"]:
    img = cv2.imread(file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 백그라운드 영역 추출
    lower = np.array([0, 0, 200], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    
    mask_background = cv2.inRange(hsv, lower, upper)
    
    # 청록색 추출
    lower = np.array([30, 0, 0], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask_target = cv2.inRange(hsv, lower, upper)
    
    # 백그라운드 반전
    mask_forground = cv2.bitwise_not(mask_background)
    # 최종 마스크 산출 
    mask = cv2.bitwise_and(mask_target, mask_forground)
    # 이미지 추출
    result = cv2.bitwise_and(img, img, mask = mask)
    # 검은색 부분 흰색으로 변경
    result[(result[...,0]==0)&(result[...,1]==0)&(result[...,2]==0)] =255
    filename = os.path.basename(file)
    save_path = f"export/{filename}"
    cv2.imwrite(save_path, result)


### train/test 데이터 뽑아보기 # 다음시간 5분까지
os.chdir("E:/cjcho_work/250929/청경채")
files = glob.glob("train/**/label.csv", recursive=True)
# type1
dfs = [pd.read_csv(file) for file in files]

# type2
dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

img_path = []
for path in df["img_name"]:
    base_dir = path.split("_")[0]
    img_path.append(f"train/{base_dir}/image/{path}")

df["img_name"] = img_path


os.makedirs("export_train", exist_ok= True)
for file in tqdm.tqdm(df["img_name"]):
    img = cv2.imread(file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 백그라운드 영역 추출
    lower = np.array([0, 0, 200], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    
    mask_background = cv2.inRange(hsv, lower, upper)
    
    # 청록색 추출
    lower = np.array([30, 0, 0], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask_target = cv2.inRange(hsv, lower, upper)
    
    # 백그라운드 반전
    mask_forground = cv2.bitwise_not(mask_background)
    # 최종 마스크 산출 
    mask = cv2.bitwise_and(mask_target, mask_forground)
    # 이미지 추출
    result = cv2.bitwise_and(img, img, mask = mask)
    # 검은색 부분 흰색으로 변경
    result[(result[...,0]==0)&(result[...,1]==0)&(result[...,2]==0)] =255
    filename = os.path.basename(file)
    save_path = f"export_train/{filename}"
    cv2.imwrite(save_path, result)

### 20분 시작

