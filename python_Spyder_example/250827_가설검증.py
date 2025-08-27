# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 09:02:59 2025

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
가설 검정
H0 : 표본 데이터의 모평균은 130일 것이다.(추정)
H1 : 모평균은 130이 아닐 것이다.
유의수준: 0.05, 양측검정
alternative="two-sided",기본값, "less","greater"
"""
stats.ttest_1samp(sample,130)
stats.ttest_1samp(sample,130,alternative="two-sided")
stats.ttest_1samp(sample,130,alternative="less")
stats.ttest_1samp(sample,130,alternative="greater")

stats.ttest_1samp(sample,130)
"""
H0: 감자의 무게의 평균은 130일 것이다.
H1: 감자의 무게의 평균은 130이 아닐 것이다.
유의수준 0.05, 양측검정

p value 0.169로 0.05보다 크므로 귀무가설을 기각하지 못함.
유의수준 0.05 하에서 감자 무게의 평균은 130이다.
np.mean(sample) # 128.4507(표본평균)
"""
"""
조금 풀어놓은 느낌
H0: 근력 운동은 성적향상 집중력 향상에 도움이 된다.
H1: 근력 운동은 성적향상에 도움이 되지 않는다.

H0 : 근력 운동 전 성적과 근력 운동 후 성적은 차이가 없다.
H1 : 근력 운동 전 성적과 근력 운동 후 성적은 차이가 있다.

H0: mu_a - mu_b = 0
H1: mu_a - mu_b != 0
"""
training_rel = pd.read_csv("./data/ch11_training_rel.csv")
training_rel
####stats.ttest_rel : 전후 바꿔도 값은 동일
###밑에 후에서 전을 뺀값가도 동일
stats.ttest_rel(training_rel["전"],training_rel["후"])
stats.ttest_rel(training_rel["후"],training_rel["전"])
"""
p value 0.04로 0.05보다 작으므로 귀무가설을 기각.
근력 운동은 성적향상에 도움이 된다.
"""
"""
H0: 근력 운동 전/후 성적의 차이가 0이다
H1: 근력 운동 전/후 성적의 차이가 0이 아니다
"""
training_rel["차"] = training_rel["후"] - training_rel["전"]
stats.ttest_1samp(training_rel["차"], 0)

training_rel["차"] = training_rel["전"] - training_rel["후"]
stats.ttest_1samp(training_rel["차"], 0)

training_ind = pd.read_csv("./data/ch11_training_ind.csv")
training_ind

"""
A: 인문계열 학생
B: 체육계열 학생
A집단의 평균이랑 B집단의 평균의 차이 비교

H0: mu_a - mu_b =0
H0: mu_a - mu_b !=0

H0: 인문계열 학생과 체육계열 학생분들이 평균성적의 차이가 없다.
H1: 인문계열 학생과 체육계열 학생분들이 평균성적의 차이가 있다.
"""
stats.ttest_ind(training_ind["A"],training_ind["B"])
stats.ttest_ind(training_ind["A"],training_ind["B"],equal_var=False)
"""
p value 0.086 으로 유의수준 0.05 보다 높아 
    귀무가설을 기각할 수 없다. 
    따라서 인문계열과 체육계열의 
    평균성적은 차이가 없다.
    혹은 차이가 있다고 할 수 없다.
"""
"""
비모수 검정
직접 구하는 부분 320 page - 324 page
wilcoxon t test와 대응
H0: 특정 처리 전/후의 중앙값의 차이가 없다.
H1: 특정 처리 하기 전과 하고 난 후의 중앙값에 차이가 있다.
"""
stats.wilcoxon(training_rel['전'], training_rel["후"])
"""
p value 가 0.0379로 유의수준 0.05보다 작아 귀무가설을 기각.
    특정 처리 전/후의 중앙값의 차이가 있다.
    > 운동 전/후의 성적의 중앙값의 차이가 존재한다.
"""
# 서수(순서를 의미)는 순서가 안바뀌는 스케일링을 한다고 해도 결과가 바뀌지 않는다.
stats.wilcoxon(training_rel['전'] - training_rel["후"] )

"""
Mann Whitney rank test
    Two sample Ttest에 대응(독립표본 T test) - 두 집단 간의 차이 비교
    두 집단간에 중앙값의 순서가 차이가 있는지 비교
327~330 page 손으로 계산
"""
stats.mannwhitneyu(training_ind["A"], training_ind["B"])

"""
H0: 인문계열 집단 A와 체육계열 집단 B의 중앙값에는 차이가 없다.
H1: 집단 A와 B의 중앙값에는 차이가 있다.
"""

"""
pvalues 0.0594로 유의수준 0.05보다 크므로 귀무가설을 기각할 수 없음.
    따라서 집단 A와 B의 중앙값에는 차이가 없다.
"""
from statsmodels.stats.descriptivestats import sign_test
sign_test(sample, 130) # 1 sample T test

# 두개 이상의 독립된 그룹에서 중앙값의 차이가 있는지 검정
group1 = np.random.randn(5)
group1
group2 = np.random.randn(5)
group2
group3 = np.random.randn(5)
group3
stats.kruskal(group1, group2, group3) 


# H0: 데이터가 정규분포를 따른다. 
# H1: 데이터가 정규분포를 따르지 않는다.
stats.shapiro(sample)

"""
등분산 검정

H0: 여러 독립된 그룹의 분산이 같다.
"""
group1 = np.random.randn(5)
group1
group2 = np.random.randn(5)
group2
group3 = np.random.randn(5)
group3
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