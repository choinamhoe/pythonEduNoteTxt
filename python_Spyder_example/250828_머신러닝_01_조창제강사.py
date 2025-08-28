# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 08:55:51 2025

@author: human
"""

"""
데이터 받는 곳

https://www.kaggle.com/datasets/vipullrathod/fish-market/data
"""
import os 
import pandas as pd
os.chdir("E:/cjcho_work/250828")

df = pd.read_csv("Fish.csv")
df["Species"].value_counts()
"""
Perch : 농어/배스
Bream : 붕어
Roach : 잉어(빨강돔)
Pike : 강꼬치고기(파이크)
Smelt : 빙어
Parkki : 빙어(핀란드나 북유럽명)
Whitefish: 백어
"""
# 30 cm 가 넘으면 도미
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import matplotlib.pyplot as plt
%matplotlib auto

plt.scatter(bream_length, bream_weight, c="red", label="도미")
plt.scatter(smelt_length, smelt_weight, c='blue', label="빙어")
plt.scatter(30, 600, c="green") # 나중에 예측할 데이터
plt.legend()

import pandas as pd 
bream_df = pd.DataFrame({"length":bream_length, "weight":bream_weight, "label":1})
smelt_df = pd.DataFrame({"length":smelt_length, "weight":smelt_weight, "label":0})
df = pd.concat([bream_df, smelt_df],axis=0, ignore_index=True)
df

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
X = df.drop("label", axis=1)
y = df["label"].values
model.fit(X,y)
model.predict([[30, 600]])

"""
KNeighborsClassifier 
K 근접 이웃 : 새로운 데이터로 예측을 하면 학습데이터 모두와 거리를 구하고
    학습데이터 중 가장 가까이 있는 K 개의 최빈값 데이터의 레이블을 예측값으로 
    사용하는 알고리즘
"""

model = KNeighborsClassifier()
train_df = df[:35]
test_df = df[35:]
model.fit(train_df.drop("label", axis=1), train_df["label"].values)
pred = model.predict(test_df.drop("label", axis=1))
sum(test_df["label"]==pred)/len(pred)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
model.fit(train_df.drop("label", axis=1), train_df["label"].values)
pred = model.predict(test_df.drop("label", axis=1))

sum(test_df["label"]==pred)/len(pred)
accuracy_score(test_df["label"], pred) # 정확도
f1_score(test_df["label"], pred) # f1 score
precision_score(test_df["label"], pred) # 정밀도
recall_score(test_df["label"], pred)# 재현율

plt.scatter(train_df["length"], train_df["weight"],c="red")
plt.scatter(test_df["length"], test_df["weight"],c='blue')

"""
데이터 누수
    : 검증데이터가 학습데이터에 포함되는 현상
    테스트 데이터를 이미 학습 했기 때문에 잘 맞출 수 밖에 없음
    다양한 이유들로 발생
"""

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(df.drop("label", axis=1), df["label"].values)
pred = model.predict(test_df.drop("label", axis=1))

accuracy_score(test_df["label"], pred) # 정확도
f1_score(test_df["label"], pred) # f1 score
precision_score(test_df["label"], pred) # 정밀도
recall_score(test_df["label"], pred)# 재현율

pred_points = [[25, 150]]
plt.scatter(df["length"], df["weight"])
plt.scatter(25, 150)
model.predict([[25,150]])

# knn은 거리기반이다보니 단위를 조심해서 사용해야 한다.
# x 축이랑 y축이랑 단위가 다르니까  단위를 통일해줘야된다.
# 데이터 스케일링을 해줄 필요가 있다.
train_df["dist"] = (train_df["length"] - 25)**2 + (train_df["weight"] - 150)**2
train_df.sort_values("dist")
# 무게의 값의 범위가 길이보다 상대적으로 커서 빙어에 가깝다고 나오는 경우가 더 많이 나옴

from sklearn.preprocessing import StandardScaler
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
scaler = StandardScaler()
scaler.fit(train_df[["length", "weight"]])

train_df[["length", "weight"]] = scaler.transform(train_df[["length", "weight"]])
test_df[["length", "weight"]] = scaler.transform(test_df[["length", "weight"]])

plt.scatter(train_df["length"], train_df["weight"])
plt.scatter(test_df["length"], test_df["weight"])

predict_points= scaler.transform([[25,150]])
plt.scatter(predict_points[0,0],predict_points[0,1])
# 스케일링을 하고 나서 거리를 계산하면 무게의 값의 범위와 길이의 값의 범위가 같아서 도미라고 나옴
train_df["dist"] = (
    train_df["length"] - predict_points[0,0])**2 + (train_df["weight"] - predict_points[0,1])**2
train_df.sort_values("dist")

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

model = DecisionTreeClassifier()
model.fit(train_df.drop("label", axis=1), train_df["label"].values)
pred = model.predict(test_df.drop("label", axis=1))
test_df["label"]
tree.plot_tree(model)

####################################
import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
plt.scatter(perch_length, perch_weight)
df = pd.DataFrame({"length":perch_length, "weight":perch_weight})
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
"""
KNN regressor: 학습데이터를 기준으로 예측하려는 지점의 값과 
가장 가까이 있는 포인트 K개를 찾고
그 K 개에 대해서 종속변수의 평균값을 최종 예측값으로 제시
"""

model = KNeighborsRegressor()
# 도미의 길이를 입력하면 무게가 나오게 모델 학습
model.fit(train_df[["length"]], train_df["weight"].values)

pred = model.predict(test_df[["length"]])
train_df.sort_values("length")
plt.scatter(train_df["length"], train_df["weight"])
plt.scatter(test_df["length"], test_df["weight"])
plt.scatter(test_df["length"], pred)

# KNN 알고리즘 특성상 가장 가까운 지점 K개의 평균을 예측값으로 사용
# 그래서 학습데이터의 종속변수의 범위를 벗어날 수 없음
# 그래서 길이를 50으로 예측했을 때 무게가 1200정도 나와야 될 것 같은데
# 1040 으로 나타나게 됨
train_df["weight"].max() # 최대값 1100
model.predict([[50]])

#################################################
# 선형회귀
plt.scatter(df["length"],df["weight"])
from sklearn.linear_model import LinearRegression
model = LinearRegression()
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
model.fit(train_df[["length"]],train_df["weight"])
pred = model.predict([[50]])
plt.scatter(df["length"],df["weight"])
plt.scatter(50, pred)
xs = np.linspace(0, 60, 200)
ys = model.predict(xs.reshape(-1,1))
plt.plot(xs,ys)

# 종속 변수를 루트 씌웠을 때
plt.scatter(df["length"],np.sqrt(df["weight"]))

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df['weight'] = np.sqrt(train_df['weight'])
model = LinearRegression()
model.fit(train_df[["length"]],train_df["weight"])
pred = model.predict([[50]])
pred = pred**2
plt.scatter(df["length"],df["weight"])
plt.scatter(50, pred)

xs = np.linspace(0, 60, 200)
ys = model.predict(xs.reshape(-1,1))
ys = ys **2
plt.plot(xs,ys)
########################################
# 길이의 제곱을 고려했을 때 
# 변수 여러개 넣어서 부를 때 다변수 다변량 다항 등으로 부름
# 변수 == 특성
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df["length_2"] = train_df["length"] **2
test_df["length_2"] = test_df["length"] **2

model = LinearRegression()
model.fit(train_df[["length","length_2"]],train_df["weight"])
pred = model.predict([[50]])
plt.scatter(df["length"],df["weight"])
plt.scatter(50,pred)

xs = np.linspace(0, 60, 200)
line_data = pd.DataFrame({"length":xs,"length_2":xs**2})
ys = model.predict(line_data)
plt.plot(xs, ys)
model.intercept_ # beta0
model.coef_ # 기울기

##########################
perch_full = pd.read_csv("https://bit.ly/perch_csv_data")
plt.scatter(perch_full["length"], perch_full[" width"])
plt.scatter(perch_full[" width"], perch_full[" height"] )
plt.scatter(perch_full["length"], perch_full[" height"] )
perch_full["weight"] = perch_weight
perch_full.columns = ["length","height", "width", "weight"]

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly.fit([[2,3]])
poly.transform([[2,3]]) # 2^0, 2^1, 3^1, 2* 3, 3^2

poly = PolynomialFeatures(degree=3) # 조합
poly.fit([[2,3]])
poly.transform([[2,3]]) # 2^0, 2^1, 3^1, 2* 3, 3^2, 2^2*3, 3^2 *2, 3^3

train_df, test_df = train_test_split(perch_full, test_size=0.1, random_state=42)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(train_df.drop("weight", axis = 1))
train_poly = poly.transform(train_df.drop("weight", axis = 1))
train_poly = pd.DataFrame(train_poly, columns=poly.get_feature_names_out())
test_poly = poly.transform(test_df.drop("weight", axis = 1))
test_poly = pd.DataFrame(test_poly, columns=poly.get_feature_names_out())


model = LinearRegression()
model.fit(train_poly, train_df["weight"])
pred = model.predict(test_poly)

plt.scatter(test_df["weight"], pred)
plt.plot(range(0,800))
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(test_df["weight"].values, pred) # 18.86

train_df.shape
test_df.shape

### 라쏘 릿지
train_df, test_df = train_test_split(perch_full, test_size=0.2, random_state=42)

scale = StandardScaler()
scale.fit(train_df.drop("weight", axis = 1))
train_scaled = scale.transform(train_df.drop("weight", axis = 1))
test_scaled = scale.transform(test_df.drop("weight", axis = 1))

from sklearn.linear_model import Ridge, Lasso, ElasticNet
model = Ridge(alpha= 1)
model.fit(train_scaled, train_df["weight"])
pred = model.predict(test_scaled)
mean_absolute_error(test_df["weight"].values, pred) # 90

#####################################################
# 로지스틱 회귀
# 이진 분류 클레스가 딱 2개인 것 분류
import pandas as pd
fish = pd.read_csv("https://bit.ly/fish_csv_data")
fish.head()
fish["Species"].value_counts()
# Bream , Smelt
df = fish.loc[fish["Species"].isin(["Bream", "Smelt"]),:].reset_index(drop=True)
df["Species"] = df["Species"].map({"Bream":1,"Smelt":0})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_df.drop("Species",axis=1), train_df["Species"])

pred = model.predict(test_df.drop("Species", axis=1))
prob = model.predict_proba(test_df.drop("Species", axis=1))

accuracy_score(pred, test_df["Species"])
model.coef_
model.intercept_
