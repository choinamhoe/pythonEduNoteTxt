# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:19:24 2025

@author: human
"""

"""
# 데이콘 Basic 고객 지원
https://dacon.io/competitions/official/236562/overview/description
"""

import os 
import pandas as pd 
os.chdir("E:\최남회\파이썬개발에대한파일모음")

train = pd.read_csv("open/train.csv")
test = pd.read_csv("open/test.csv")
train
test

train.info()
desc = train.describe()
desc
"""
ID : 고유아이디
age: 고객 나이(18~65)
gender : 고객 성별(tr : M:56% F:44%)
tenure : 고객이 서비스를 이용한 총 기간(월) 1~60개월(수치량)
frequent : 고객의 서비스 이용일 1~30일(수치일)
payment_interval : 고객의 결제 지연일 1~30일(수치일)
subscription_type : 고객의 서비스 등급(member, plus, vip)
contract_length : 고객의 서비스 계약 기간(30, 90, 360)
after_interaction : 고객이 최근 서비스 이용으로부터 경과한 기간
  > 안쓴지 얼마나 됬는지 1~30(일)
support_needs : 고객의 지원 필요도 (0: 낮음, 1: 중간, 2: 높음)
"""

import matplotlib.pyplot as plt
%matplotlib auto
train["gender"].value_counts()/train.shape[0] #남/녀비율
train["tenure"].value_counts() #수치형
train["frequent"]
train["payment_interval"]
train["subscription_type"].value_counts()
train["contract_length"].value_counts()
train["after_interaction"].max()

train["age_group"] = pd.cut(train["age"], bins = range(10,81,10), right=False)
temp = train.groupby(["age_group","gender"]) # size().reset_index()
import seaborn as sns
#성별 나이대별 분포
sns.countplot(x="age_group",hue="gender", data =train)

train["type"] = "train"
test["type"] = "test"

vis_df = pd.concat([train,test], ignore_index=True)
vis_df["age_group"] = pd.cut(vis_df["age"], bins=range(10,81,10),right=False)
vis_df
sns.countplot(x="age_group",hue="gender", data =vis_df)

### 학습/검증 데이터의 나이대 분포
vis_df.pivot(columns = "type", values = "age")
plt.hist(
    vis_df.pivot(columns = "type", values = "age"),
    bins=30,
    density=True,
    label = ["test", "train"]
    )
plt.legend()
## 성별 데이터의 나이 분포
vis_df.pivot(columns = "gender", values = "age")
plt.hist(
    vis_df.pivot(columns = "gender", values = "age"),
    bins=30,
    density=True,
    label = ["F", "M"]
    )
plt.legend()


## 나이대별 지원 필요도(0,1,2) 분포
# type1
vis_df.groupby(["age_group","support_needs"]).size()
temp = vis_df.groupby(["age_group","support_needs"]).size()
temp = temp.reset_index()
temp.index = temp["age_group"]
temp.drop("age_group",axis = 1).pivot(columns="support_needs",values=0)

#type2
temp = vis_df.groupby(["age_group","support_needs"]).size()
temp.unstack(fill_value=0).plot(kind="barh")

## 나이대별 고객등급 분포
temp = vis_df.groupby(["age_group","subscription_type"]).size()
temp.unstack(fill_value=0).plot(kind="barh")

## 고객등급별 지원 필요도(0,1,2)의 분포
sns.countplot(
    x="subscription_type",
    hue="support_needs",
    palette="Pastel1",
    data =vis_df)

## 등급별 지원 필요도(0,1,2)의 분포
# 25분
# 등급별(subscription_type), 지원필요도(support_needs)
temp = vis_df.groupby(["support_needs","subscription_type"]).size()
temp.unstack(fill_value=0).plot(kind="barh")

temp = vis_df.groupby(["subscription_type","support_needs"]).size()
temp.unstack(fill_value=0).plot(kind="barh")

## 나이대별 지원필요도 분포
temp = vis_df.groupby(["age_group","support_needs"]).size()
temp = temp.unstack(fill_value=0)
temp.sum(axis=1)
ratio = temp.div(temp.sum(axis=1),axis=0)
ax = ratio.plot(
    kind = "barh",
    stacked=True,
    colormap="Pastel1"
    )
0.508/2, 0.508+0.186/2, 0.508+0.186+0.304/2   
ax.text(
        0.3,
        0,
        "50%",
        va = "center",
        ha = "center",
        color="black",
        fontsize=9
        )

for i,row in enumerate(ratio.index):
    for col in ratio.columns:
        raise

col = 0.0
value = ratio.loc[row,col]
before_value = value
x_loc = value/2
ax.text(
        x_loc,
        i,
        f"{value*100:.1f}%",
        va = "center",
        ha = "center"
        )

col = 1.0
value = ratio.loc[row,col]
x_loc = value/2 + before_value
before_value = before_value + value
ax.text(
        x_loc,
        i,
        f"{value*100:.1f}%",
        va = "center",
        ha = "center"
        )

col = 2.0
value = ratio.loc[row,col]
x_loc = value/2 + before_value
before_value = before_value + value
ax.text(
        x_loc,
        i,
        f"{value*100:.1f}%",
        va = "center",
        ha = "center"
        )
###################
temp = vis_df.groupby(["age_group", "support_needs"]).size()
temp = temp.unstack(fill_value=0)
temp.sum(axis=1)
ratio = temp.div(temp.sum(axis=1),axis=0)
ax = ratio.plot(
    kind="barh",
    stacked=True,
    colormap="Pastel1"
    )

for i, row in enumerate(ratio.index):
    before_value = 0 
    for col in ratio.columns:
        value = ratio.loc[row,col]
        x_loc = value/2 + before_value
        before_value += value
        ax.text(
                x_loc,
                i,
                f"{value*100:.1f}%",
                va = "center",
                ha = "center"
                )
"""
col = 2.0
value = ratio.loc[row,col]
x_loc = value/2 + before_value
before_value = before_value + value
ax.text(
        x_loc,
        i,
        f"{value*100:.1f}%",
        va = "center",
        ha = "center"
        )
"""

## 16분 for 문 으로 작업 

## 35분 시작

# 나이대별 최종
"""
after_interaction : 고객이 최근 서비스 이용으로부터 경과한 기간
  > 안쓴지 얼마나 됬는지   1~30(일)
"""
sns.boxplot(
    x="age_group",y="after_interaction", hue="support_needs" ,
    data=vis_df)
vis_df.info()

## 오래 사용한 고객일 수록 지원 필요도가 낮은지
# tenure 고객이 서비스를 이용한 총 기간
# 큰 차이는 나지 않지만 오래 사용한 고객일 수록 지원 필요도가 낮다
sns.boxplot(x="support_needs",y="tenure",data=vis_df, palette="Pastel1")

## 계약기간별 지원 필요도 분포
# 계약기간이 30, 90, 360 
# 계약기간이 짧으면 지원이 많이 필요한지 

sns.countplot(
    x="contract_length", hue="support_needs", data=vis_df, palette="Pastel1")