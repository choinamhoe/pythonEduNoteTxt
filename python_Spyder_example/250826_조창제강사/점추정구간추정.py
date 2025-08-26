# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:57:08 2025

@author: human
"""

import os
import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
%matplotlib auto
os.chdir("C:/Users/human/Downloads/source/python_stat_sample-master")
df = pd.read_csv("./data/ch4_scores400.csv")
df
"""
모평균: 모집단의 평균
모분산: 모집단의 분산
"""
mu = df.mean().values # 모 평균
std = df.std(ddof=0).values # 모 분산

xs = np.arange(101)
rv = stats.norm(mu, std)
plt.plot(xs, rv.pdf(xs), color="gray")
plt.hist(df["score"], bins=100, density=True, color="blue", alpha=0.3)

"""
점 추정:특정 하나의 값을 추정
    표본을 통해 모수를 추정하는 것
    어떻게 추정해야 좋은 추정치인가 라는 것을 고민한 것
구간 추정: 범위를 추정 하는 것
"""

n = 20 
sample = np.random.RandomState(42).choice(df["score"], n)
sample.shape
# 10000번 반복해서 df["score"] 값을 랜덤하게 20개씩 추출해라
n_samples = 10000
samples = np.random.RandomState(42).choice(
    df["score"], (n_samples,n))
samples.shape
samples[0]
samples[1]

for i in range(5):
    s_mean = np.mean(samples[i])
    print(f"{i+1} 번째 표본평균: {s_mean:0.3f}")
    
# 각각의 20개에대한 평균 1만개
sample_means = np.mean(samples, axis = 1)

plt.scatter(range(10000), sample_means)
plt.hist(sample_means, bins=100, density=True)

### 불편 추정량
"""
수많은 표본들 속에서 어떠한 점추정량이 좋은 점추정량인가

편향: 데이터가 한쪽으로 치우쳐 짐을 의미
불편: 편향의 반대의미, 데이터가 치우쳐지지 않았다.
    중심 경향 척도
"""
# 표본 평균에 평균 혹은 표본 평균에 기대값은 
# 불편 추정량
np.mean(sample_means)

# 모평균 값이랑 표본평균 값이 같을 수록 좋은 값
# 모수의 값이랑 통계량 값이 동일할수록 좋은 값
"""
표본평균의 기대값은 불편 추정량이면서 일치 추정량 이라 할 수 있음
"""
np.mean(sample_means)

"""
점추정 - 어떤 모수가 좋은 모수인가
모수가 어느 범위까지 신뢰할수 있는가  -> 구간추정
"""

"""
점추정 결과 > 표본평균의 기대값
"""
# 272p 
# 분포를 알고 있다는 가정 -> 유의수준 얼마까지 신뢰할 수 있다.
# 정규분포(???,모분산) 구간 추정


# 표본평균 N(mu, sigmia^2/n)
# 표준오차: 표준 편차의 추정량

"""
표본 평균의 기대값

표준화 
(x - mu)/sigma 

표준 정규분포 형태를 띔

# 유의수준 0.05 -> 95% 통계적으로 신뢰 하겠다.
P(z_0.975 <= (x_bar-mu)/np.sqrt(sig^2/n) <= z_0.025) = 0.95
영역 -> 신뢰구간

신뢰구간(Confidence Interval): 점 추정량을 통계적으로 유의수준 하에서 
    어느 범위까지 신뢰할 것인지 나타내는 구간
    
표본 평균의 기대값을 유의수준 0.05 기준으로 
    z_0.975부터 z_0.025까지 통계적으로 신뢰하겠다. 

"""
s_mean = np.mean(sample_means)
# s_var = np.mean(np.var(samples, axis=1, ddof=1))
s_var  = std**2
n = 20
rv = stats.norm(0, 1)
s_mean - rv.isf(0.025)*np.sqrt(s_var/n)
s_mean - rv.isf(0.925)*np.sqrt(s_var/n)


"""page 297
귀무 가설: "차이가 없다" "효과가 없다" 는 형태, 모수 = 0
대립 가설: 주장하고 싶은 가설, "차이가 있다"는 형태로 나타남 모수 !=0, <0, >0

유의하다: 우연이 아니라 어떤 의미가 있는 것을 말한다.
    통계적으로 의미가 있다. 표본 데이터상으로 의미가 있다.

기각역: 귀무가설이 기각되는 영역
유의수준: 기각역에 들어가는 확률 혹은 면적
임계값: 유의수준에서 가지는 x 값

통계량: 표본 데이터 기준으로 산출되는 값(모수에 대응되는 값)
p값: 통계량 기준으로 바깥의 영역의 면적(확률 값처럼 나온다)
""" 
# 감자 무게에 대한 표본 평균 
# z 통계량
s_mean = 128.451
(s_mean - 130)/np.sqrt(9/14) # 통계량 값 -1.931

rv = stats.norm(0, 1)
rv.isf(0.95) # 기각역 수치값이 -1.644 

"""
표준정규분포를 따르는 데이터에서 
유의수준 0.05 하에서 기각역이 -1.644이고
통계량 값이 -1.931 이므로 기각역보다 통계량 값이 작으므로
귀무가설을 기각
통계적으로 평균이 130이라고 할 수 없다(통계적인 근거가 부족하다.)
"""

rv.isf(0.975) # 양측 검정 기준 기각역 -1.9599

"""
95% 신뢰 하겠다 -> 유의수준 0.05

1종오류: 귀무가설이 참인데 귀무가설을 기각하는 경우
    (MU 가 130인데 130이 아니라 하는 경우)
2종오류: 대립가설이 참인데, 귀무가설을 채택하는 경우 
    (Mu 가 130이 아닌데, 130이라고 하는 경우)
이해해 두는 것이 좋음
"""
# 모평균 130이라 추정, 모분산 9라고 가정
# 감자 샘플데이터
df = pd.read_csv("./data/ch11_potato.csv")
sample = df["무게"]
s_var = sample.var() # 표본 분산

s_mean = np.mean(sample) # 표본 평균
rv = stats.norm(0, 1)  # 표준 정규 분포
interval = rv.interval(0.95) # 양측 검정에 대한 기각역

n = sample.shape[0] # 표본 개수
z = (s_mean - 130) / np.sqrt(9/n) # z 통계량 값 산출
# 기각역을 벗어나는지 테스트  True면 벗어나지 않음
(interval[0] <=z)&(z <=interval[1]) 

rv.cdf(z)*2 # z 통계량에 대한 p value 값



"""
모분산 추정
"""

sample = df["무게"]
s_var = np.var(sample, ddof= 1)
n = sample.shape[0]
rv = stats.chi2(df = n-1)
interval = rv.interval(0.95)

y = (n-1)*s_var /9
(interval[0]<=y) & (y<= interval[1])

""" 요건 내일"""
y<rv.isf(0.5)
(1-rv.cdf(y))*2


