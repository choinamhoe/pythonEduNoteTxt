# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:42:39 2025

@author: human
"""

import os
import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
% matplotlib auto
# 회귀분석 검정

"""
등분산 검정 

H0: 여러 독립된 그룹의 분산이 같다.
"""
group1 = np.random.randn(5)
group2 = np.random.randn(5)
group3 = np.random.randn(5)
stats.bartlett(group1, group2, group3) 
stats.levene(group1, group2, group3)

""" 
정규성 검정
H0: 데이터가 정규분포를 따른다 가정
    or 
H0: 데이터가 특정 분포를 따른다 가정
"""
stats.shapiro(sample)
stats.anderson(sample, dist= "norm") 
# norm, expon, logistic, gumbel, gumbel_l, weibull_min, extreme1

## 회귀분석
df = pd.read_csv("./data/ch12_scores_reg.csv")
df
"""
학생 20명

수학 점수
쪽지시험(quiz)
기말점수(final_test)
수면시간(sleep_time)
교통수단(버스, 자전거, 걷기)
"""

plt.scatter(
    df["quiz"], df["final_test"],
    c=df["school_method"].map({
        "bicycle":"red",
        "bus":"blue",
        "walk":"green"
        }))

plt.scatter(
    df["sleep_time"], df["final_test"],
    c=df["school_method"].map({
        "bicycle":"red",
        "bus":"blue",
        "walk":"green"
        }))

import statsmodels.formula.api as smf
df.head(1)
model = smf.ols(
    "final_test ~ quiz + sleep_time", df).fit()
model.summary()

import statsmodels.api as sm
X = df[["quiz","sleep_time"]]
X["Intercept"] = 1
y = df["final_test"].values
model = sm.OLS( y, X).fit()
model.summary()

## 각자 모델 진단하기 17분까지 진행
"""
등분산 검정이랑 독립성 검정은 회귀 모델 새우기 전에 했다고 가정
F test
H0: 모든 회귀계수는 0이다.
p value 가 유의수준 0.05 보다 작으므로 귀무가설을 기각 -> 회귀 계수가 적어도 하나는 의미가 있다.

T test
H0: quiz 변수에 대한 bata 값은 0이다.
H1: coefficient 값은 0이 아니다.
> quiz는 회귀 계수가 의미가 있다.

beta 0 는 회귀 계수가 통계적으로 의미가 있지는 않다.
"""

import statsmodels.api as sm
X = df[["quiz"]]
X["Intercept"] = 1
y = df["final_test"].values
model = sm.OLS( y, X).fit()
model.summary()

# 2차원 시각화
plt.scatter(df["quiz"], df["final_test"])
xs = np.linspace(0, 10, 200)
ys = 23.6995 + 6.5537 * xs
plt.plot(xs, ys)

# 3차원 시각화
import plotly.graph_objects as go
quiz_range = np.linspace(X['quiz'].min(), X['quiz'].max(), 30)
sleep_range = np.linspace(X['sleep_time'].min(), X['sleep_time'].max(), 30)
quiz_grid, sleep_grid = np.meshgrid(quiz_range, sleep_range)

Z = model.params['Intercept'] + model.params['quiz']*quiz_grid + model.params['sleep_time']*sleep_grid

# 3D 산점도
scatter = go.Scatter3d(
    x=X['quiz'],
    y=X['sleep_time'],
    z=y,
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Observed'
)

# 회귀 평면
surface = go.Surface(
    x=quiz_grid,
    y=sleep_grid,
    z=Z,
    colorscale='Blues',
    opacity=0.5,
    name='Regression Plane'
)

# 레이아웃
layout = go.Layout(
    title='3D Regression Plane',
    scene=dict(
        xaxis_title='Quiz',
        yaxis_title='Sleep Time',
        zaxis_title='Final Test'
    )
)

fig = go.Figure(data=[scatter, surface], layout=layout)
fig.write_html("E:/cjcho_work/250827/3d_regression.html")


############

plt.scatter(
    df["quiz"], df["final_test"],
    c=df["school_method"].map({
        "bicycle":"red",
        "bus":"blue",
        "walk":"green"
        }))
xs = np.linspace(0, 10, 200)
ys = 23.6995 + 6.5537 * xs
plt.plot(xs, ys)

import statsmodels.api as sm
dummies = pd.get_dummies(
    df["school_method"],dtype=np.int32)

X = df[["quiz","sleep_time"]]
X = pd.concat([X,dummies],axis=1)
X["Intercept"] = 1
y = df["final_test"].values
model = sm.OLS( y, X).fit()
model.summary()



X = df[["quiz"]]
X = pd.concat([X,dummies],axis=1)
X["Intercept"] = 1
y = df["final_test"].values
model = sm.OLS( y, X).fit()
model.summary()

plt.scatter(
    df["quiz"], df["final_test"],
    c=df["school_method"].map({
        "bicycle":"red",
        "bus":"blue",
        "walk":"green"
        }))
xs = np.linspace(0, 10, 200)
ys1 = 18.8207 + 6.5537 * xs + 8.1879
ys2 = 18.8207 + 6.5537 * xs + 7.7231
ys3 = 18.8207 + 6.5537 * xs + 2.9097
plt.plot(xs, ys1, 
         color= "red", label="bicycle")
plt.plot(xs, ys2, 
         color= "blue", label="bus")
plt.plot(xs, ys3, 
         color= "green", label="walk")
plt.legend()


