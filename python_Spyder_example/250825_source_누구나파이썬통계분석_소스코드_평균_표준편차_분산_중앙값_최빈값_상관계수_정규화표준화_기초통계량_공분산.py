# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:13:57 2025

@author: human
"""

import os
os.chdir("E:\최남회\source_누구나파이썬통계분석_소스코드\python_stat_sample-master")
import pandas as pd
df = pd.read_csv("./data/ch1_sport_test.csv",
                 index_col = "학생번호")
df
df["악력"]

import numpy as np 
df = pd.read_csv(
    "./data/ch2_scores_em.csv",
    index_col="student number")

df
df.head()
######### 평균
scores = df.loc[:10, "english"].values
scores

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
scores
scores = sorted(scores)
scores
len(scores) #전체길이
len(scores) / 2 #전체길이의 절반
scores[int(len(scores)/2)]

scores = df.loc[:9, "english"].values
scores
scores = sorted(scores)
scores
len(scores)/2
(scores[4] + scores[5])/2

df.loc[:10,"english"].median()
#####
#최빈값 : 빈도가 가장 높은 값
pd.Series([1,1,1,2,2,3]).mode()
pd.Series([1,1,1,2,2,3]).value_counts()

#### 분산, 표준편차
import matplotlib.pyplot as plt
%matplotlib auto

plt.scatter(range(10), scores)

#편차
#각각의 값이 평균 값이랑
#얼마나 떨어져 있는지 나타내는 값
scores = df.loc[:10, "english"].values
scores
#scores = sorted(scores)
#scores
score_mean = np.mean(scores) # 평균
score_dev = scores - score_mean #편차
score_dev

#편차
plt.scatter(range(10),scores)
plt.hlines(score_mean,0,10)
plt.scatter(range(10),socre_dev)

scores = df.loc[:10, "english"].values
scores

result = 0
for i in scores:
    print(i)
    result = result + i
result
n = len(scores)
n

result_mean = result/n
result_mean

sum(scores)/len(scores)

## for문써서 표준편차
result = list()
for i in scores:
    print(i)
    print(result_mean)
   # print(i - result_mean)
    result.append(i - result_mean)
result
np.array(result)

##표본분산
pd.Series(scores).var(ddof=0) #모분산
pd.Series(scores).var() #표본분산

##for문써서 모분산
result = 0
for i in scores:
    print(i)
    cal = (i - result_mean)**2
    result = result + cal
result
result/len(scores) #모분산
result/len(scores) 

sorted(scores)[1]
sorted(scores)[2]
#분위수
q3 = np.quantile(scores, 0.75)
q3
q1 = np.quantile(scores, 0.25)
q1
iqr = (q3 - q1)
iqr
#최대값,최소값
np.min(scores)
np.max(scores)
#범위
np.max(scores) - np.min(scores)
pd.Series(scores).min()
pd.Series(scores).max()
pd.Series(scores).std()/pd.Series(scores).mean()
q3+1.5*iqr,0.8,1.2
q1-1.5*iqr,0.8,1.2

plt.scatter([1]*10, scores)
plt.hlines(q3,0.8,1.2)
plt.hlines(q1,0.8,1.2)
plt.hlines(q3+1.5*iqr,0.8,1.2)
plt.hlines(q1-1.5*iqr,0.8,1.2)

####################################################
#정규화 표준화
#데이터 범위나 분포를 바꾸는 작업을 수행하는 것을 의미
#표준화는 평균이 0, 표준편차가 1로 데이터 분포가 
#변경 될 수 있게 하는 작업을 의미
####################################################
scores = df.loc[:10, "english"].values
scores
scores - np.mean(scores)

#표준화 시각화
#그림 1
plt.scatter(range(10), scores) #전체 데이터
plt.scatter(range(10), scores - np.mean(scores))
#그림 2
plt.scatter(range(10), scores - np.mean(scores))
plt.scatter(range(10), scores - np.mean(scores)/np.std(scores))

# N(mu, sigma) -> Z(0, 1)
# ddof는 의존적으로 선택되는 숫자의 갯수
# 자유도 개념의 반대 degress of Freedos = ddof
# numpy는 기본값이 ddof = 0
# pandas는 pd.Series(scores).var() 는 표본분산 즉,ddof = 1인 값
np.std(scores)
np.std(scores, ddof=0)
np.std(scores, ddof=1)

#기초통계량
df["english"].describe()
df.shape
res = plt.hist(df["english"],bins=10)
res
res_df = pd.DataFrame(res[:2]).T
res_df.columns=["빈도","범위"]
res_df

#상자그림
plt.boxplot(scores)
#########################################start
# (#########################################start~end)
#같이 실행 할 것
sub_df = df.loc[:10, 
                ["english","mathematics"]].copy()
sub_df.index = list("ABCDEFGHIJ")

plt.scatter(
    sub_df["english"], 
    sub_df["mathematics"])

for plot_id, (eng, math) in zip(
        sub_df.index, sub_df.values):
    plt.text(eng, math, plot_id)
#########################################end
    
# 공분산 Covariance
# 범위가 - Inf ~ Inf
# 수식상으로 분산인데 컬럼 2개를 고려한 것
df.cov()
df.loc[:, 
                ["english","mathematics"]].cov()
df["english"].var()
# 상관계수 correlation coefficient
# 범위가 -1 ~ 1
# 수식상으로 범위가 -1~1이되게 정규화 한 것
df.loc[:, 
                ["english","mathematics"]].corr()

#########################################start
# (#########################################start~end)
#같이 실행 할 것
plot_df = df.reset_index().copy()

plt.scatter(
    plot_df["english"],
    plot_df["mathematics"])

for plot_id, eng, math in plot_df.values:
    plt.text(eng, math, plot_id)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(
    plot_df[["english"]], 
    plot_df["mathematics"].values)

y_pred = model.predict(plot_df[["english"]])
plt.scatter(
    plot_df["english"],
    y_pred)
#########################################end    
import statsmodels.api as sm 
plot_df["const"] = 1
X = plot_df.drop(
    ["student number", "mathematics"],axis=1)
Y = plot_df["mathematics"].values
X.shape, Y.shape
model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())
y_pred = results.predict(X) # 값이 이상하게 나옴
y_pred

plt.scatter(
    plot_df["english"],
    plot_df["mathematics"])

for plot_id, eng, math in plot_df[
        ["student number", "english","mathematics"]].values:
    plt.text(eng, math, plot_id)


plt.scatter(X["english"],y_pred)

y_pred = 42.6013 + 0.6214 * plot_df["english"]
plt.scatter(X["english"],y_pred)

anscombe_data = np.load("./data/ch3_anscombe.npy")
anscombe_data .shape
anscombe_data = np.load("./data/ch3_anscombe.npy")
anscombe_data .shape
stats_df = pd.DataFrame(index=['X_mean', 'X_variance', 'Y_mean',
                               'Y_variance', 'X&Y_correlation',
                               'X&Y_regression line'])
for i, data in enumerate(anscombe_data):
    dataX = data[:, 0]
    dataY = data[:, 1]
    poly_fit = np.polyfit(dataX, dataY, 1)
    stats_df[f'data{i+1}'] =\
        [f'{np.mean(dataX):.2f}',
         f'{np.var(dataX):.2f}',
         f'{np.mean(dataY):.2f}',
         f'{np.var(dataY):.2f}',
         f'{np.corrcoef(dataX, dataY)[0, 1]:.2f}',
         f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x']
stats_df
stats_df.shape
#########################################
df = pd.read_csv("./data/ch4_scores400.csv")
# replace 옵션은 중복 추출 가능 하게 해줌
# size 는 뽑는 개수 기본값은 1
# p 기재 해주면 확률을 따로 줄 수 있음
np.random.choice([1, 2, 3], replace = False, size = 3)
np.random.choice([1, 2, 3], replace = True, size = 3)
np.random.choice([1, 2, 3], replace = True, size = 20, p = [0.1, 0.4, 0.5])
#########################################
#df를 랜덤하게 300개 형을 train_df로 나머지 100개를 test_df로 만들기
df.shape[0]
np.random.choice(range(400), size = 10)

total = list(range(df.shape[0]))
total
train_idx = np.random.choice(total, size = 300)
train_idx
test_idx = list(set(total) - set(train_idx))
test_idx
train_df = df.loc[train_idx,:]
test_df = df.loc[test_idx,:]

import sklearn
train_df, test_df = sklearn.model_selection.train_test_split(
    df, test_size = 100)
train_df.shape
test_df.shape

###################
# 주사위가 불공정한 확률을 가짐

sum(range(1,7))
dice = [i for i in range(1,7)]
prob = [i/21 for i in range(1,7)]

# 100번 시행
sample = np.random.choice(dice , size = 100, p = prob)

summary_df = pd.Series(sample).value_counts()
summary_df = summary_df.sort_index().reset_index()
summary_df["relative frequency"] = summary_df["count"]/100
summary_df["prob"] = prob

(pd.Series(sample).value_counts()/100).sort_index()
prob

plt.scatter(range(6), summary_df["relative frequency"])
plt.scatter(range(6), summary_df["prob"])
plt.bar(range(1,7), summary_df["relative frequency"])
plt.bar(range(1,7), summary_df["prob"],width=0.1)

#####
plt.hist(
    df["score"], bins = 100, range=(0, 100), density = True)


new_sample = np.random.choice(df["score"], 10000)
plt.hist(
    new_sample, bins = 100, range=(0, 100), density = True,alpha=0.5)

sample_mean = df["score"].mean()
plt.vlines(sample_mean, 0, 0.05)
