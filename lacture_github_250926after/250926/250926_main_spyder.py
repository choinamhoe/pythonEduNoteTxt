import cv2 , tqdm, os
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt 
import glob

%matplotlib auto

os.chdir("E:/choinamhoe/lacture_github_250926after/250926")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# train data
# id, digit, letter
img = train.iloc[318,3:].values
img = img.reshape(28,28).astype(float)

# train 데이터의 이미지를 train 폴더에 저장하는 작업
os.makedirs("train",exist_ok=True)
value = train.values[0]
# type 1
data_id = value[0]
label = value[1]
text = value[2]
# type2
# data_id, label, text = value[0:3]
img = value[3:].reshape(28,28).astype(float)
img = cv2.resize(img, (224,224))
filename = f"id_{data_id}_text_{text}_label_{label}.png"
save_path = f"train/{filename}"
cv2.imwrite(save_path, img)
### 40분 까지 for 문으로 
for value in tqdm.tqdm(train.values):
    data_id, label, text = value[0:3]
    img = value[3:].reshape(28,28).astype(float)
    img = cv2.resize(img, (224,224))
    filename = f"id_{data_id}_text_{text}_label_{label}.png"
    save_path = f"train/{filename}"
    cv2.imwrite(save_path, img)

os.makedirs("test",exist_ok=True)
for value in tqdm.tqdm(test.values):
    data_id, text = value[0:2]
    img = value[2:].reshape(28,28).astype(float)
    img = cv2.resize(img, (224,224))
    filename = f"id_{data_id}_text_{text}_label_none.png"
    save_path = f"test/{filename}"
    cv2.imwrite(save_path, img)
    
## 45분 시작
tr_files = glob.glob("train/*")
train = pd.DataFrame({"path":tr_files })

te_files = glob.glob("test/*")
test = pd.DataFrame({"path":te_files })

# batch 사이즈만큼 
class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, ??):
        ???
    def __len__(self):
        # 전체길이/batch수
        ??
    def __getitem__(self, index):
        # X(batch, width, height, channel), y(batch, label)
        ???
    def on_epoch_end(self):
        # data suffle
        ???
# 다음시간 5분 
index = 0
batch_size = 4
df = train.copy()

st = index*batch_size
ed = (index+1)*batch_size

paths = df["path"].values[st:ed]
file_path = paths[0]

# file_path 입력하면 x, y 나오게 함수 만들기 12분
####
y = int(file_path.split("_")[-1].split(".")[0])
x = cv2.imread(file_path)
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
_y = np.zeros(10)
_y[y] = 1
y = _y
def preprocessing(file_path):
    y = int(file_path.split("_")[-1].split(".")[0])
    x = cv2.imread(file_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    _y = np.zeros(10)
    _y[y] = 1
    y = _y
    return x, y

x,y=preprocessing(file_path)
x.shape,y.shape

"""
paths 만큼 x랑 y 가 preprocessing 함수가 동작해서
(batch, width, height), (batch, label) 형태로 나오게 코드 작성
20분
"""
# type 1
x_list = []
y_list = []
for file_path in paths:
    x, y= preprocessing(file_path)
    x_list.append(x)
    y_list.append(y)
np.array(x_list).shape
np.array(y_list).shape
## 
# (batch, w, h, c), (batch, label)

# type 2
np.array([cv2.imread(file_path) for file_path in paths]).shape
int(paths[0].split("_")[-1].split(".")[0])

## y 숫자로 추출
y_temp = [int(i.split("_")[-1].split(".")[0]) for i in paths]
## 더미화 진행
y = np.zeros((len(y_temp),10))
for idx, value in enumerate(y_temp):
    y[idx,value] = 1