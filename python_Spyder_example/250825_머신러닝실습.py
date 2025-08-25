# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:17:43 2025

@author: human
"""
import matplotlib.pyplot as plt
%matplotlib auto


#iris 데이터 불러와서
#Train/Test 로 나누고
#1.Petal Length로 Sepal Length 예측하기(회귀모델)
#30분에 풀이
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
iris
iris.columns
print(iris["DESCR"])
df = pd.DataFrame(
    data = iris.data, 
    columns = iris.feature_names)
df
df.columns
"""
Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)'],
      dtype='object')
"""
df.loc[:, "target"] = [
    iris.target_names[i] 
    for i in iris.target]
df
df.columns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train,test = train_test_split(df, test_size= 0.1)
train.shape, test.shape

train.columns
model = LinearRegression()
model.fit(train[["petal length (cm)"]], train["sepal length (cm)"])

y_pred = model.predict(test[["petal length (cm)"]])
y_true = test["sepal length (cm)"]
sum((y_true-y_pred)**2)
import matplotlib.pyplot as plt
%matplotlib auto
plt.scatter(y_pred, y_true)

########## setosa는 꽃잎의 길이가 증가함에 따라 꽃받침의 길이의 영향을 거의 안받음
########## virginica,versicolor는 꽃잎의 길이가 증가함에 따라 꽃받침의 길이가 증가
########## y 절편 값이 다른거 같음
train["target"].value_counts()
colors = {"setosa":"red", "virginica":"blue","versicolor":"green"}
plt.scatter(
    train["petal length (cm)"], train["sepal length (cm)"], 
    c = train["target"].map(colors))
plt.plot(range(0,9))


#2.모든 변수 자유로이 사용해서
#Sepal Length 예측하기
#다음시간 05분에 풀이
train1 = train[train["target"].isin(["setosa"])] # setosa data
train2 = train[~train["target"].isin(["setosa"])] # virginica, versicolor
train2["is_virginica"] = (train2["target"]=="virginica")+0

test1 = test[test["target"].isin(["setosa"])] # setosa data
test2 = test[~test["target"].isin(["setosa"])] # virginica, versicolor
test2["is_virginica"] = (test2["target"]=="virginica")+0

model1 = LinearRegression()
model2 = LinearRegression()
model1.fit(train1[["petal length (cm)"]], train1["sepal length (cm)"])
model2.fit(
    train2[["petal length (cm)","is_virginica"]], 
    train2["sepal length (cm)"])

y_pred2 = model2.predict(test2[["petal length (cm)","is_virginica"]])
y_pred1 = model1.predict(test1[["petal length (cm)"]])
y_true2 = test2["petal length (cm)"]
y_true1 = test1["petal length (cm)"]

y_pred = np.concatenate([y_pred1,y_pred2])
y_true = np.concatenate([y_true1,y_true2])
sum((y_true-y_pred)**2)

plt.scatter(y_pred, y_true)
plt.scatter(y_pred2, y_true2)
plt.scatter(y_pred1, y_true1)
