# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:40:26 2025

@author: human
"""

"""
연속 확률 분포 : 간격이 실수 간격
정규분포,지수분포,카이제곱분포,t분포,F분포,(감마분포)
(감밸분포, 와이블 분포, 플렛쳇 분포)
"""

"""
정규분포 : 데이터가 자연스럽게 샘플링 됐을 때 나타나는 데이터의 분포
가우스 분포라고도 하고 정규성,정상성 등으로 이름이 불림

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

f = warp(170,5)
f(30)
# 평균이 170이고 표준편차가 5, 우연히 만난 남학생 키가 165~175
from scipy import integrate
result , err = integrate.quad(warp(170,5), 165, 175)
result

#모의고사 평균이 70, 표준편차가 8, 점수의 범위가 54~86 사이의 확률은
fun = warp(70, 8)
res, err =  integrate.quad(fun, 54, 86)
res #95.4 확률로 54~86 범위 안에 존재한다.

#평균이 170이고 표준편차가 5, 우연히 만난 남학생 키가 165~175
from scipy import stats
rv = stats.norm(170,5)
rv.cdf(175)-rv.cdf(165)

import matplotlib.pyplot as plt
%matplotlib auto
#평균이 170이고 표준편차가 5인 정규분포에서 x가 150~190 까지 시각화한 결과
plt.plot(range(150,190),rv.pdf(range(150,190)))
plt.vlines(175,0,0.09) # x가 175인 위치 시각화
#rv.isf(0.05) 오른쪽에서부터 영역 값이 0.05가 되는 x의 위치를 찾는 함수
#plt.vlines(rv.isf(0.05),0,0.09,color="red")
x_fill = np.linspace(150, 175,200)
y_fill = rv.pdf(x_fill)
plt.fill_between(
    x_fill, y_fill, 
    color ="skyblue", alpha=0.5, label="cdf(175)")

# rv.isf(0.05) 오른쪽에서부터 영역 값이 0.05가 되는 x 의 위치를 찾는 함수
#plt.vlines(rv.isf(0.05),0, 0.09, color="red")
x_fill = np.linspace(rv.isf(0.05), 190,200)
y_fill = rv.pdf(x_fill)
plt.fill_between(
    x_fill, y_fill, 
    color ="red", alpha=0.5, label="alpha")
plt.legend()

rv = stats.norm(0,1)
x_values = np.linspace(-4, 4, 100)
plt.plot(x_values,rv.pdf(x_values),
         color="red",linestyle="--", label="N(0,1)")

rv = stats.norm(0,4)
x_values = np.linspace(-4, 4, 100)
plt.plot(x_values,rv.pdf(x_values),
         color="blue",linestyle=":", label="N(0,4)")

rv = stats.norm(1,1)
x_values = np.linspace(-4, 4, 100)
plt.plot(x_values,rv.pdf(x_values),
         color="green",linestyle="dotted", label="N(1,1)")
plt.legend()

"""
지수분포: 어떤 사건이 발생하는 간격이 따르는 분포
"""
#하루 평균 2건 사건 발생 3일 이내에 사고가 발생활 확률

def warp(lam):
    def fun(x):
       return lam * np.exp(-lam*x)
    return fun

result , err = integrate.quad(warp(2), 0, 3)
result

#한시간 평균 10번 사건의 발생, 1분 이내에 사건이 발생할 확률
result , err = integrate.quad(warp(10), 0, 1/60)
result

"""
카이제곱 분포:
    분산의 구간측정이나 독립성 검정시 사용
"""
from scipy import stats
n = 10 # 자유도
rv = stats.norm(0,1)
z_samples = rv.rvs((n,1000000))
z_samples.shape

ch2_sample = np.sum(z_samples**2,axis=0)
#표본 데이터 생성
ch2_sample

plt.hist(ch2_sample, bins = 100, density=True, alpha=0.5, label="ch2")
xs = np.linspace(0, 50, 100)
rv = stats.chi2(10)
ys = rv.pdf(xs)
plt.plot(xs, ys, label="chi2(10)",color="gray")
plt.legend()

"""
x~t(n)
t분포:
    z가 표준정규분포를 따르고
    Y가 자유도 n인 카이제곱 분포 따름

t = Z/ sqrt(Y/n)
분자는 평균의 카이를 표준화 한 값
분모는 분산 개념으로 표준화 한 값
1. 주로 평균에 차이가 있는지 검증 할 때 사용
    -one sample t test: 하나의 그룹이 평균이 얼마가 맞는지 검증 mu - 80 = 0
    -two sample t(independent) test :
        두 그룹의 평균의 차이가 있는지 검증 mu1 - mu2 = 0
    -paried t test:
        한그룹의 사전 사후 차이가 있는지 검증 mu_a - mu_b = 0
        (예 약물 처리전과 처리 후 효과가 있는지)
2. 회귀 분석에서 각 변수의 기울기가 0인지 검증할 때 사용
    -H0: beta0 = 0

특징:
    1. 정규분포처럼 좌우대칭 형태의 분포
    2. 표준정규분포 대비 꼬리부분이 두꺼움
    3. 자유도가 높아질수록 표준정규분포에 근사
"""

n = 10
rv1 = stats.norm(0,1) #표준정규분포
rv2 = stats.chi2(n) #카이제곱분포

sample_size = 1000000
z_sample = rv1.rvs(sample_size)
chi2_sample = rv2.rvs(sample_size)

t_sample = z_sample/np.sqrt(chi2_sample/n)

plt.hist(t_sample, bins = 100)

"""
z test 는 모 표준편차를 알아야되니 비현실적 
 > 분산(표준편차)을 추정하자

카이제곱 분포: 자료가 n 개 있을 때 분산들이 따르는 분포

t test 는 
카이제곱 분포를 활용하여 
자유도를 알면 모 표준편차를 사용하지 않아도 
평균을 추정할 수 있게 만든 분포

F 분포 
통계량: 카이제곱의 분산 비
집단 사이의 분산이 동일한가 비교 할 때 사용 
"""
rv1 = stats.norm(0,1)
xs1 = np.linspace(-4,4,100)
ys1 = rv1.pdf(xs1)

rv2 = stats.t(3)
ys2 = rv2.pdf(xs1)

plt.plot(xs1,ys1, label="N(0,1)")
plt.plot(xs1,ys2, label="t(3)")


rv3 = stats.t(100)
ys3 = rv3.pdf(xs1)
plt.plot(xs1,ys3, label="t(100)")
plt.legend()

"""
카이제곱 분포: 자료가 n 개 있을 때 분산들이 따르는 분포
F 분포 : 두 그룹이 존재하고 각각의 그룹의 카이제곱 분포의 분산에 대한 비율

모수 자유도 값을 2개 입력(각각의 그룹에 대한 자유도)
F분포의 범위는 0이상의 실수
"""

rv = stats.f(5, 10)
xs = np.linspace(0,6,100)
ys = rv.pdf(xs)
plt.plot(xs,ys, label="f(5,10)")


rv = stats.f(10, 5)
xs = np.linspace(0,6,100)
ys = rv.pdf(xs)
plt.plot(xs,ys, label="f(10,5)")
plt.legend()

"""
교재특징(누구나 파이썬 통계분석 교재)
분포에 대해서 친절하게 용어로 정리 되어 있어서 다시 한번 보면 좋아 보입니다.
단어 선택은 좋지 못함..
"""
"""
중심극한정리: 원래 데이터 분포가 어떤 분포를 따르든지 관계없이
관측수가 많아지면 표본평균은 정규분포에 가까워진다.
"""
#람다가 3인 포아송 분포 따르는 데이터 1만개에 대한 평균 n세트는
#표본평균이 정규분포를 따른다.
rv = stats.poisson(3)

n = 1000
sample = 10000
pos_sample = rv.rvs((n,sample))
sample_mean = np.mean(pos_sample, axis = 0)

rv2 = stats.norm(3,np.sqrt(3/n))
np.linspace(rv2.isf(0.999), rv2.isf(0.001),100)
ys = rv2.pdf(xs)
plt.hist(sample_mean, bins=100, density=True)
plt.plot(xs,ys,color="red")