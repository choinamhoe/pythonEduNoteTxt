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
# 9분시작

# 블러 처리
aug = A.Blur(blur_limit=50, p=1)
res = aug(image=img)["image"]
plt.imshow(res)

# 홀수만 입력 가능
aug = A.GaussianBlur(blur_limit=49, p=1)
res = aug(image=img)["image"]
plt.imshow(res)

# -0.5  ~ 0.5
"""
밝기와 대비 
https://natadioka1023.tistory.com/111
20분 시작
"""
aug = A.RandomBrightness(limit=0.5,p=1)
res = aug(image=img)["image"]
plt.imshow(res)
dir(A)

aug = A.RandomContrast(limit=0.5, p=1)
res = aug(image=img)["image"]
plt.imshow(res)

"""
# HSV 색조 채도 명도
# https://hyunhp.tistory.com/682
26분 시작
"""
hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hsv_img.shape
hsv_img[...,0].min(), hsv_img[...,0].max()
new_img = img.copy()
# hsv_img[...,2]는 이미지 내 명도 값
# 명도(밝기)가 220보다 크면 값을 255로 변경
new_img[hsv_img[...,2]>220]=255 
#plt.imshow(new_img)

aug = A.HueSaturationValue(
    hue_shift_limit=20,
    sat_shift_limit=20,
    val_shift_limit=20,p=1)
res = aug(image=img)["image"]
plt.imshow(res)

### 34분시작
aug = A.ISONoise(
    color_shift=(0.01,0.05),
    intensity=(0.5,5), p=1)
res = aug(image=img)["image"]
plt.imshow(res)

### 다양한 augmentation 동시에 적용
augmentations = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ]
transform = A.Compose(
    augmentations)

res = transform(image=img)["image"]
plt.imshow(res)

CoarseDropout = A.augmentations.dropout.coarse_dropout.CoarseDropout
# 이미지에 랜덤하게 0값을 가지는 사각형 
aug = CoarseDropout(
    max_holes=8,  #최대 8개 사각형 생성
    max_height=20, # 최대 높이 20
    max_width=20,  # 최대 너비 20
    fill_value=0, # 채울 색상 
    p=1)
res = aug(image=img)["image"]
plt.imshow(res)

### 
# 갑자기 어그멘테이션을 추가하고 싶을 때 
new_augmentations = [
    *augmentations,
    aug]
transform = A.Compose(new_augmentations)
res = transform(image=img)["image"]
plt.imshow(res)

### 내가 어그멘테이션을 만들어서 쓰고 싶을 때
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np

class MyHorizontalFlip(ImageOnlyTransform):
    def __init__(self, p=1.0, always_apply=False):
        super().__init__(always_apply, p)
        self.p = p

    def apply(self, image, **params):
        # 확률보다 낮으면 동작하고 높으면 동작하지 않게(랜덤하게)
        if np.random.rand() < self.p:
            image = image[:,::-1] # 수평으로 동작 
        return image

aug2 = MyHorizontalFlip()
aug2(image=img)["image"]

new_augmentations = [
    *augmentations,
    aug,
    aug2
    ]
transform = A.Compose(new_augmentations)
res = transform(image=img)["image"]
plt.imshow(res)

