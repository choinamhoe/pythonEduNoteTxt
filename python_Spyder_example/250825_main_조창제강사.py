# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 11:05:12 2025

@author: human
"""
import os 
os.chdir(
    "C:/Users/human/Downloads/source/python_stat_sample-master")
import pandas as pd

df = pd.read_csv(
    "./data/ch1_sport_test.csv",
    index_col = "학생번호")
df

import numpy as np 
df = pd.read_csv(
    "./data/ch2_scores_em.csv",
    index_col="student number")

df.head()
######### 평균
scores = df.loc[:10, "english"].values

# type1
sum(scores)/len(scores)
# type2
df.loc[:10,"english"].mean()

########## 중앙값
# 전체 자료중에 정 중앙에 위치한 값
# 정 중앙의 위치가 딱 떨어지면 해당값
# 정 중앙의 위치가 딱 떨어지지 않으면 
# 근방의 2값의 평균

scores = df.loc[:10, "english"].values
scores = sorted(scores)
len(scores) # 전체 길이
len(scores)/ 2 # 전체 길이의 절반
scores[int(len(scores)/2)]

scores = df.loc[:9, "english"].values
scores = sorted(scores)
len(scores)/2
(scores[4] + scores[5])/2

df.loc[:10,"english"].median()


#####
# 최빈값 : 빈도가 가장 높은 값
pd.Series([1,1,1,2,2,3]).mode()
pd.Series([1,1,1,2,2,3]).value_counts()

#### 분산, 표준편차
import matplotlib.pyplot as plt 
%matplotlib auto



# 편차
# 각각의 값이 평균 값이랑 
# 얼마나 떨어져 있는지 나타내는 값
scores = df.loc[:10, "english"].values
score_mean = np.mean(scores) # 평균
score_dev = scores - score_mean #편차

plt.scatter(range(10),scores)
plt.hlines(score_mean,0,10)
plt.scatter(range(10),score_dev)
