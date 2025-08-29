# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 13:53:40 2025

@author: human
"""

import numpy as np 
import matplotlib.pyplot as plt 
import cv2
%matplotlib auto
fruits = np.load("C:/Users/human/Downloads/fruits_300.npy")
fruits.shape # 흑백 이미지 300장 -> 과일

plt.imshow(fruits[100], cmap="gray")
# 25분 시작

save_path = "E:/cjcho_work/250829/imgs"
for i, img in enumerate(fruits):
    save_file_path = f"{save_path}/x_{i}.png"
    cv2.imwrite(save_file_path, img)
img.shape
save_file_path
# 30분 시작

## 이미지를 비지도 학습을 통해 분리해내는게 목적!
plt.imshow(fruits[100], cmap="gray_r")
plt.imshow(fruits[100], cmap="gray")

apple = fruits[:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:].reshape(-1, 100*100)
# 원상복구 banana = banana.reshape(100,100,100)

apple.mean(axis=1).shape # (100,)
plt.hist(apple.mean(axis=1))
plt.hist(pineapple.mean(axis=1))
plt.hist(banana.mean(axis=1))

apple.mean(axis=0).shape# (10000,)
plt.imshow(fruits[:100].mean(axis=0)) # 사과
plt.imshow(fruits[100:200].mean(axis=0)) # 파인애플
plt.imshow(fruits[200:].mean(axis=0)) # 바나나

plt.bar(range(10000), apple.mean(axis=0))
plt.bar(range(10000), pineapple.mean(axis=0))
plt.bar(range(10000), banana.mean(axis=0))

plt.imshow(fruits[:100].mean(axis=0), cmap="gray_r") # 사과
plt.imshow(fruits[100:200].mean(axis=0), cmap="gray_r") # 파인애플
plt.imshow(fruits[200:].mean(axis=0), cmap="gray_r") # 바나나



apple_mean = apple.mean(axis=0).reshape(100, 100)
pineapple_mean = pineapple.mean(axis=0).reshape(100, 100)
banana_mean = banana.mean(axis=0).reshape(100, 100)
fruits .shape, apple_mean.shape
abs_diff = np.abs(fruits - apple_mean)
plt.imshow(fruits[0], cmap="gray_r")
plt.imshow(apple_mean, cmap="gray_r")
plt.imshow(np.abs(fruits[0]-apple_mean), cmap="gray_r")
plt.imshow(np.abs(fruits[100]-apple_mean), cmap="gray_r")
plt.imshow(np.abs(fruits[200]-apple_mean), cmap="gray_r")

abs_diff[:100].mean()
abs_diff[100:200].mean()
abs_diff[200:].mean()

import os
main_path = "E:/cjcho_work/250829/results"
for i, (img_diff,img) in enumerate(zip(abs_diff,fruits)):
    if img_diff.mean()<20:
        save_path = f"{main_path}/apple"
        save_file = f"{save_path}/x_{i}.png"
    elif img_diff.mean()<40:
        save_path = f"{main_path}/pineapple"
        save_file = f"{save_path}/x_{i}.png"
    else:
        save_path = f"{main_path}/banana"
        save_file = f"{save_path}/x_{i}.png"
    os.makedirs(save_path,exist_ok=True)
    cv2.imwrite(save_file, img)

################
# 20분 이어서 진행
# 320p
fruits = np.load("C:/Users/human/Downloads/fruits_300.npy")
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(fruits_2d)
model.labels_

import os
main_path = "E:/cjcho_work/250829/results_2"
for i, (img,label) in enumerate(zip(fruits,model.labels_)):
    save_path = f"{main_path}/x_{label}"
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/x_{i}.png"
    cv2.imwrite(save_file, img)

############################### 
# 4분
# https://velog.io/@jhlee508/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-K-%ED%8F%89%EA%B7%A0K-Means-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98 

###############################
# https://rpubs.com/qkdrk777777/511938
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib auto
n = 1000
x1 = np.random.RandomState(42).normal(0, 1, n)
y1 = np.random.RandomState(42).normal(0, 1, n)

r2 = 7 + np.random.RandomState(42).normal(0, 1, n)
t2 = np.random.RandomState(42).uniform(0, 2*np.pi, n)

x2 = r2*np.cos(t2)
y2 = r2*np.sin(t2)

r= 4
x = np.concatenate([x2,x2/r])
y = np.concatenate([y2,y2/r])

plt.scatter(x, y)

data = np.concatenate([x[:,np.newaxis],y[:,np.newaxis]], axis=1)
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
model = KMeans(n_clusters=2, random_state=42)
model.fit(data)

plt.scatter(x, y, c=model.labels_)
plt.scatter(*model.cluster_centers_.T)

model = DBSCAN(eps= 1.3 , min_samples =2)
model.fit(data)
plt.scatter(x, y, c=model.labels_)

# 02 분 시작
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster 
from matplotlib import pyplot as plt
# 계층적 군집 
# https://todayisbetterthanyesterday.tistory.com/61 

# data, method ="single", "complete" etc., metric = "euclidean"
linked = linkage(data, 'single')  # 바텀업 방식
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
# criterion 자르는 기준 , distance: 거리, maxclust: 최대 군집 개수
# t는 기준값
cut_tree = fcluster(linked, t=1.4, criterion="distance")
cut_tree # label
pd.Series(cut_tree).value_counts()
cut_tree = fcluster(linked, t=3, criterion="maxclust")
pd.Series(cut_tree).value_counts()

#################################
# 차원축소
# PCA: 공분산이나 상관계수 활용해서 차원축소하는 방식
#  선형적인 특성만 고려가 가능하다 
# tSNE: 비선형적인 특성을 고려해서 차원 축소하는 방식
#  주로 시각화 할 때 많이 사용

from sklearn.decomposition import PCA
fruits = np.load("C:/Users/human/Downloads/fruits_300.npy")
fruits.shape
fruits_2d = fruits.reshape(-1, 100*100)

pca = PCA()
pca.fit(fruits_2d)
pca.components_.shape
# 스크리 플롯: 급격히 떨어지는 형태에 그래프
# kmeans: k개를 어떻게 구하지? 스크리플롯을 활용
plt.plot(pca.explained_variance_) 

pca = PCA(n_components=30)
pca.fit(fruits_2d)
pca.transform(fruits_2d).shape

