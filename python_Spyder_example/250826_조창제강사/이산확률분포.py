# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 08:59:13 2025

@author: human
"""

"""
베르누이 분포: 동전 던지기와 같이 1번의 시행과 같이
특정 확률로 성공 혹은 실패가 나오는 데이터가 따르는 분포

**시행을 1번 한다. 0과 1로 나타난다가 중요 부분**
ex1) 동전을 던져서 앞면이 나올 확률
ex2) 주사위를 한번 굴려서 6이 나오지 않을 확률
"""
def bern_f(p, x):
    """
    베르누이 확률함수
    """
    if x in [0,1]:
        return p**x  * (1-p) **(1-x)
    else:
        return 0
# 동전을 던져 앞면이 나올 확률
bern_f(0.5, 1)
# 주사위를 한번 굴려서 6이 나오지 않을 확률
bern_f(1/6, 0)

"""
pmf: 확률 질량 함수 
cdf: 누적 밀도 함수
"""
from scipy import stats
# 동전을 던져 앞면이 나올 확률
rv = stats.bernoulli(0.5)
rv.pmf(1)
# 주사위를 한번 굴려서 6이 나오지 않을 확률
rv = stats.bernoulli(1/6)
rv.pmf(0)
rv.cdf(1)

"""
이항분포: 성공확률이 p인 베르누이 시행을 n번 했을 때,
 성공 횟수가 따르는 분포
"""
from scipy.special import comb
# comb: 조합 nCr = n!/r!(n-r)!
def fun(n, p, x):
    if x in range(n):
        return comb(n, x) * p **x * (1-p)**(n-x)
    else:
        return 0
# 동전을 10번 던졌을 때 앞면이 3번 나올 확률
fun(10,1/2, 3)
# 주사위를 4번 던져서 6이 나오지 않을 확률
fun(4, 1/6, 0)
# 주사위를 4번 던져서 2나 6이 2번 확률
fun(4, 2/6, 2)

# 패키지 활용
rv = stats.binom(10, 1/2)
rv.pmf(3)

# 누적 밀도함수는 확률밀도함수들 더한 값
rv.pmf(0)+rv.pmf(1)+ rv.pmf(2)
rv.cdf(2)

import matplotlib.pyplot as plt
%matplotlib auto

fig, ax = plt.subplots(figsize=(10, 6))
# n = 10 , p = 0.3
rv = stats.binom(10, 0.3)
probs = rv.pmf(range(10))
ax.plot(range(10), probs, color = "red", label = "binom(10, 0.3)")

# n = 10 , p = 0.5
rv = stats.binom(10, 0.5)
probs = rv.pmf(range(10))
ax.plot(range(10), probs, color = "green", label = "binom(10, 0.5)")

# n = 10 , p = 0.7
rv = stats.binom(10, 0.7)
probs = rv.pmf(range(10))
ax.plot(range(10), probs, color = "blue", label = "binom(10, 0.7)")
ax.legend()

# for문 활용
fig, ax = plt.subplots(figsize=(10, 6))
n = 10
for prob, color in zip([0.3, 0.5, 0.7], ["red","green", "blue"]):
    rv = stats.binom(n, prob)
    probs = rv.pmf(range(n))
    ax.plot(range(n), probs, color = color, label = f"binom({n}, {prob})")
ax.legend()

"""
기하분포: 처음 성공할 때까지 반복한 시행횟수가 따르는 분포
"""

def fun(p, x):
    if x>=1:
        return p*(1-p)**(x-1)
    return 0

# 동전을 던져 다섯 번째 처음으로 앞면이 나올 확률
fun(1/2, 5)
# 주사위를 던져 3번째 처음으로 6이 나올 확률
fun(1/6, 3) # fun(6이나올확률, n번째)
# 주사위를 던져 6번 째 처음으로 짝수가 나올 확률
fun(3/6, 6)

# 동전을 던져 5번째 처음으로 앞면이 나올 확률
rv = stats.geom(1/2)
rv.pmf(5)

rv = stats.geom(1/6)
probs = rv.pmf(range(1, 20))
plt.plot(probs)
plt.bar(range(0,19), probs)
rv.mean()# 분포에 대한 평균 
rv.var()# 분포에 대한 분산
plt.vlines(rv.mean(), 0, 0.2, color="black")

"""
포아송 분포: 단위 시간 당 발생하는 사건의 건수가 따르는 확률 분포
"""
from scipy.special import factorial 
import numpy as np
def fun(lam, x):
    if x>=0:
        return lam**x / factorial(x) * np.exp(-lam)
    else:
        return 0

# 하루에 평균 2건 교통사고가 발생하는 지역에서 교통사고가 발생하지 않을 확률
fun(2,0)

# 한 시간에 평균 10번 사건이 발생, 15번 발생할 확률
fun(10, 15)


# 분포에 대한 평균 분산 계산 (p 114, p 172) 이산, 연속 확률변수
# 45분까지 자유롭게 읽어보기

