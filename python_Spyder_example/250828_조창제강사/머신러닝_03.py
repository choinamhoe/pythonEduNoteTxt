# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:29:36 2025

@author: human
"""
import numpy as np 
import pandas as pd
wine = pd.read_csv("https://bit.ly/wine_csv_data")

# target: class
# 독립변수: alcohol(도수), sugar(당도), pH(산성도?)
wine["class"].value_counts()
wine["class"] = wine["class"].astype(np.int32)
wine.isna().sum() # 결측 유무 확인
wine.describe()
import matplotlib.pyplot as plt
%matplotlib auto
plt.scatter(range(6497), wine["sugar"])

from sklearn.model_selection import train_test_split
train, test = train_test_split(
    wine, test_size = 0.2, random_state=42)

#train_x, train_y, test_x, test_y = train_test_split(
#    df_x, df_y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train.drop('class', axis = 1), train["class"])
pred = model.predict(test.drop("class", axis=1))

from sklearn.metrics import accuracy_score
accuracy_score(test["class"], pred) # 0.784615

### 스케일링 적용 했을 때
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(train.drop("class",axis = 1))
train_x = scale.transform(train.drop("class",axis = 1))
test_x = scale.transform(test.drop("class",axis = 1))

model.fit(train_x , train["class"])
pred = model.predict(test_x )
accuracy_score(test["class"], pred) # 0.789230

from sklearn.tree import DecisionTreeClassifier, plot_tree
model = DecisionTreeClassifier(random_state=42)
model.fit(train.drop('class', axis = 1), train["class"])
pred = model.predict(test.drop("class", axis=1))
accuracy_score(test["class"], pred) # 0.8769
plot_tree(model)

model = DecisionTreeClassifier(random_state=42,max_depth=2)
model.fit(train.drop('class', axis = 1), train["class"])
pred = model.predict(test.drop("class", axis=1))
accuracy_score(test["class"], pred) # 0.816
plot_tree(
    model, filled=True, 
    feature_names=["alcohol","sugar", "pH"], fontsize=14,
    class_names=["1","0"])
train["class"].value_counts()
## 16분 진행

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
"""
ada Boost
ng boost
cat boost(카테고리)
hist GBM(Histogram Gradient Boosting Model)
light GBM
xgboost

boosting 모델 특징
 > 잘 못맞추는 데이터에 가중치를 주어 
 다음모델에서 잘 맞추는 형태로 학습하다보니
 과적합하기 쉽다는 특징이 있음
 > 하이퍼 파라미터에 따른 성능의 차이가 큼 
 > 잘 설정하면 예측성능이 우수
"""
model = XGBClassifier(random_state =42, )
model.fit(train.drop('class', axis = 1), train["class"])
pred = model.predict(test.drop("class", axis=1))
accuracy_score(test["class"], pred) # 0.816

importance = model.feature_importances_
for f, imp in zip(['alcohol', 'sugar', 'pH'], importance):
    print(f"{f}: {imp:.3f}")
plt.bar(['alcohol', 'sugar', 'pH'], importance)
plt.ylabel("Feature Importance")
plt.show()

model = LGBMClassifier(random_state=42)
model.fit(train.drop('class', axis = 1), train["class"])
pred = model.predict(test.drop("class", axis=1))
accuracy_score(test["class"], pred) # 0.816
