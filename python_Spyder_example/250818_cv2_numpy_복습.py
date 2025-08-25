# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 08:54:34 2025

@author: human
"""

# pip install numpy==1.26 opencv-python
#이게 설치가 안될 경우 pip install numpy opencv-python 로 설치 할것
#cv2.imread할때 한글 폴더로 된 경로는 읽지 못함.
#이미지경로를 영문으로 된 폴더로 호출할 것
import cv2
import numpy as np 
#이미지 보여줄때 쓰는 패키지
import matplotlib.pyplot as plt
files = [
    "E:/choinamhoe/images/img1.jpg",
    "E:/choinamhoe/images/img2.jpg",
    "E:/choinamhoe/images/img3.jpg"
    ]
cv2.imread(files[0]).shape #이미지 경로는 영문으로 되어야 호출이 가능
cv2.imread(files[1]).shape #이미지 경로는 영문으로 되어야 호출이 가능
cv2.imread(files[2]).shape #이미지 경로는 영문으로 되어야 호출이 가능
imgs = []
for file in files:
    img = cv2.imread(file)
    new_img = cv2.resize(img, (512,512))
    imgs.append(new_img)
np.array(imgs).shape

arrs = np.array(imgs)
plt.imshow(arrs[0]) # 이럴 경우 rgb색상이 아닌 bgr로 나옴.그래서 rgb색상으로 나오게 하기 위해 [:,:,::-1] 추가 
plt.imshow(arrs[1]) # 이럴 경우 rgb색상이 아닌 bgr로 나옴.그래서 rgb색상으로 나오게 하기 위해 [:,:,::-1] 추가
plt.imshow(arrs[2]) # 이럴 경우 rgb색상이 아닌 bgr로 나옴.그래서 rgb색상으로 나오게 하기 위해 [:,:,::-1] 추가

plt.imshow(arrs[0][:,:,::-1])
plt.imshow(arrs[1][:,:,::-1]) # 이럴 경우 rgb색상이 아닌 bgr로 나옴.그래서 rgb색상으로 나오게 하기 위해 [:,:,::-1] 추가
plt.imshow(arrs[2][:,:,::-1]) # 이럴 경우 rgb색상이 아닌 bgr로 나옴.그래서 rgb색상으로 나오게 하기 위해 [:,:,::-1] 추가

#concatrate
imgs_concatenate = []
for file in files:
    #height, width, channel(Blue,Green,Red)
    img = cv2.imread(file)
    #cv2.resize(원하는 width,원하는 height)
    new_img = cv2.resize(img, (512,512))
    imgs_concatenate.append(new_img[np.newaxis])
np.concatenate(imgs_concatenate).shape

arrs_concatenate = np.concatenate(imgs_concatenate)
plt.imshow(arrs_concatenate[0][:,:,::-1])
plt.imshow(arrs_concatenate[1][:,:,::-1]) # 이럴 경우 rgb색상이 아닌 bgr로 나옴.그래서 rgb색상으로 나오게 하기 위해 [:,:,::-1] 추가
plt.imshow(arrs_concatenate[2][:,:,::-1]) # 이럴 경우 rgb색상이 아닌 bgr로 나옴.그래서 rgb색상으로 나오게 하기 위해 [:,:,::-1] 추가





