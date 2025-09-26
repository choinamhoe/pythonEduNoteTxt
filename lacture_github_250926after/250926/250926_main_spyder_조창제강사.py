"""
데이터 증강(augmentation) 원본 데이터의 변형을 가하여 
인공지능 모델 학습 시 다양성을 조금 더 확보하는 기술

데이터가 충분하지 못할 때 주로 활용

https://velog.io/@gyurili/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A6%9D%EA%B0%95Data-Augmentation-%EA%B8%B0%EB%B2%95-%EC%A0%95%EB%A6%AC
"""

# 20분 시작
import cv2
import os
os.chdir("E:/cjcho_work/250926")
import matplotlib.pyplot as plt
%matplotlib auto
img = cv2.imread("sample.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

## 28분 시작
#pip install albumentations numpy==1.23 scipy

"""
conda create -n temp python==3.10
conda activate temp
pip install tensorflow==2.8 numpy==1.23 albumentations matplotlib opencv-python spyder
"""

import albumentations as A
dir(A)
# 수직 반전
aug = A.HorizontalFlip(p=0.5)
res = aug(image=img)["image"]
plt.imshow(res)

# 수평 반전
aug = A.VerticalFlip(p=0.5)
res = aug(image=img)["image"]
plt.imshow(res)

# 랜덤하게 height width 만큼 짤라서 보여줌
aug = A.RandomCrop(height=300, width=300, p=1)
res = aug(image=img)["image"]
plt.imshow(res)

# 사진 정 중앙에서 300 300
aug = A.CenterCrop(height=300, width=300, p=1)
res = aug(image=img)["image"]
plt.imshow(res)
# -45~45도 랜덤하게 회전 p는 회전이 발생할 확률
aug = A.Rotate(limit=45, p=1)
res = aug(image=img)["image"]
plt.imshow(res)

