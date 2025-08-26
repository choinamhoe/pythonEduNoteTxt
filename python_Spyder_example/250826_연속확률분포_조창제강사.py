# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:45:27 2025

@author: human
"""

"""
연속 확률 분포: 간격이 실수 간격

정규분포, 지수분포, 카이제곱분포, t분포, F분포, (감마분포)
(감벨분포, 와이블 분포, 플렛쳇 분포)
"""

"""
정규분포: 데이터가 자연스럽게 샘플링 됬을 때 나타나는 데이터의 분포
가우스 분포라고도 하고 정규성, 정상성 등으로 이름이 불림

평균이랑 분산을 요구
x ~ N(mu, sigma**2) 
> 내가 가진 데이터(x)는  평균이 mu 표준편차가 sigma인 정규분포를 따른다 라는 의미

# 표준화 -> 데이터를 표준정규분포를 따르게 변환 하는 것
표준 정규 분포는 평균이 0 표준편차가 1인 정규분포를 의미
z = X-mu / sigma 
z ~ N(0, 1)
"""
import numpy as np 

def warp(mu, sigma):
    print(mu, sigma)
    def fun(x):
        result1 = (np.sqrt(2*np.pi)*sigma)
        result2 = -((x-mu)**2)/(2*sigma**2)
        return 1/result1 * np.exp(result2)    
    return fun

f = warp(170, 5)
f(165)
# 평균이 170이고 표준편차가 5, 우연히 만난 남학생 키가 165~175
from scipy import integrate
result, err = integrate.quad(warp(170,5), 165, 175)
result

# 모의고사/ 평균이 70 표준편차가 8, 점수의 범위가 54~86 사이일 확률은 
fun = warp(70, 8)
res, err = integrate.quad(fun, 54, 86)
res # 95.4% 확률로 54~86 범위 안에 존재한다.

# 평균이 170이고 표준편차가 5, 우연히 만난 남학생 키가 165~175
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib auto

rv = stats.norm(170, 5)
rv.cdf(175)-rv.cdf(165)
# 평균이 170이고 표준편차가 5인 정규분포에서 x가 150~ 190 까지 시각화한 결과
plt.plot(range(150,190), rv.pdf(range(150,190))) 
plt.vlines(175,0,0.09) # x 가 175 인 위치 시각화
x_fill = np.linspace(150, 175,200)
y_fill = rv.pdf(x_fill)
plt.fill_between(
    x_fill, y_fill, 
    color ="skyblue", alpha=0.5, label="cdf(175)")

# rv.isf(0.05) 오른쪽에서부터 영역 값이 0.05가 되는 x 의 위치를 찾는 함수
plt.vlines(rv.isf(0.05),0, 0.09, color="red")
x_fill = np.linspace(rv.isf(0.05), 190,200)
y_fill = rv.pdf(x_fill)
plt.fill_between(
    x_fill, y_fill, 
    color ="red", alpha=0.5, label="alpha")
plt.legend()
"""
정규분포는 평균을 중심으로 좌우 대칭인 종형 형태의 분포를 의미
아래는 평균과 표준편차에 따른 시각화 예시
"""
rv = stats.norm(0, 1)
x_values = np.linspace(-4, 4,100)
plt.plot(x_values, rv.pdf(x_values),
         color = "red", linestyle="--", label="N(0,1)") 

rv = stats.norm(0, 4)
x_values = np.linspace(-4, 4,100)
plt.plot(x_values, rv.pdf(x_values),
         color = "blue", linestyle=":", label="N(0,4)") 

rv = stats.norm(1, 1)
x_values = np.linspace(-4, 4,100)
plt.plot(x_values, rv.pdf(x_values),
         color = "green", linestyle="dotted", label="N(1,1)") 
plt.legend()

"""
지수분포: 어떤 사건이 발생하는 간격이 따르는 분포
"""
# 하루 평균 2건 사건 발생 3일 이내에 사고가 발생할 확률 
def warp(lam):
    def fun(x):
        return lam * np.exp(-lam*x)
    return fun

result, err = integrate.quad(warp(2), 0, 3)
result
# 한시간 평균 10번 사건이 발생, 1분이내에 사건이 발생할 확률
result, err = integrate.quad(warp(10), 0, 1/60)
result

"""
카이제곱 분포: 
    독립성 검정 시 사용, 분산의 구간 추정
"""
# 자유도가 10인 카이제곱 분포 하에서 100만개의 샘플을 생성
# 표준정규분포를 따르는 데이터를 기반으로 
# 샘플링한 것들이 카이제곱 분포를 따른다라는 것을 보여주기 위한 예제 같음
from scipy import stats
n = 10  # 자유도 
rv = stats.norm(0, 1)
z_samples = rv.rvs((n, 1000000))
z_samples.shape

ch2_sample = np.sum(z_samples**2, axis=0)
# 표본 데이터 생성 
ch2_sample

plt.hist(ch2_sample, bins = 100, density=True, alpha=0.5, label="ch2")
xs = np.linspace(0,50, 100)
rv = stats.chi2(10)
ys = rv.pdf(xs)
plt.plot(xs, ys, label="chi2(10)",color="gray")
plt.legend()

"""
x ~ t(n)
t 분포:
    z가 표준정규분포 따르고
    Y가 자유도 n인 카이제곱 분포 따름
    
t = Z/ sqrt(Y/n)
분자는 평균의 카이를 표준화 한 값
분모는 분산 개념으로 표준화 한 값
1. 주로 평균에 차이가 있는지 검증 할 때 사용
   - one sample t test: 하나의 그룹이 평균이 얼마가 맞는지 검증 mu -80 =0
   - two sample t(independent) test : 
       두 그룹의 평균이 차이가 있는지 검증 mu1 -mu2 =0
   - paired t test: 
       한 그룹의 사전 사후 차이가 있는지 검증 mu_a - mu_b = 0
       (예 약물 처리전과 처리 후 효과가 있는지)
2. 회귀 분석에서 각 변수의 기울기가 0인지 검증할 때 활용
    - H0: beta0 = 0

특징: 
    1. 정규분포처럼 좌우대칭 형태의 분포
    2. 표준정규분포 대비 꼬리부분이 두꺼움
    3. 자유도가 높아질수록 표준정규분포에 근사
"""

n = 10
rv1 = stats.norm(0,1) # 표준정규분포
rv2 = stats.chi2(n) # 카이제곱분포

sample_size = 1000000
z_sample = rv1.rvs(sample_size)
chi2_sample = rv2.rvs(sample_size)

t_sample = z_sample/np.sqrt(chi2_sample/n)

plt.hist(t_sample, bins = 100)
