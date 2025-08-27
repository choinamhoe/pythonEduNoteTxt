# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:45:11 2025

@author: human
"""

import os
import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
%matplotlib auto
os.chdir("E:\최남회\source_누구나파이썬통계분석_소스코드\python_stat_sample-master")

df = pd.read_csv("./data/ch11_potato.csv")
df
sample = df["무게"].values
sample
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
import statsmodels.formula.api as smf
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
import statsmodels.formula.api as smf
df.head(1)
model1 = smf.ols("final_test ~ quiz + sleep_time", df).fit()
model1.summary()

import statsmodels.api as sm
X = df[["quiz","sleep_time"]]
X["Intercept"] = 1
y = df["final_test"].values
model2 = sm.OLS(y,X).fit()
model2.summary()

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
fig.write_html("E:/최남회/python_Spyder_example/250827/3d_regression.html")
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


"""
더미화
범주(카테고리형 데이터) 형태의 컬럼을
0과 1로 더미화 하여 모델에 고려하는 것을 의미

회귀분석에서는
기울기가 동일하고 카테고리별로 절편 값이 다른 것 같을 때 사용
"""
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

"""
결정 계수 : 모델이 학습데이터의 변동성을
얼마나 설명하는지 나타내는 수치
0.68이라고 하면 학습데이터의 종속변수의
변동성을 68% 설명한다.

수정결정계수 : 결정계수를 독립변수의 개수로 나누어준 수치
컬럼 개수가 다른 회귀 모델 간의 비교에 사용
직접적으로 해석할수 없음
Adj. R^2의 값이 0.68 이라고 해도 변수 개수로 나눠준 값이기 때문에
68% 해석한다라고 할 수 없음.

따라서 모델 비교에는 수정결정계수
모델 해석에는 결정계수를 사용해야 함

결정계수를 R^2 라고 부르는 이유는 
독립변수를 1개만 고려했을 때 
상관계수의 제곱값이 
R^2 값이랑 동일해서 결정계수를 R^2라고 함
"""