# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 10:12:50 2025

@author: human
"""
import numpy as np 
import pandas as pd
wine = pd.read_csv("https://bit.ly/wine_csv_data")
wine["class"] = wine["class"].astype(np.int32)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Hold Out 
train, test = train_test_split(wine, test_size = 0.1)
model = LogisticRegression()
model.fit(train.drop("class",axis=1), train["class"])
pred = model.predict(test.drop("class",axis = 1))
res = accuracy_score(test["class"], pred)
res

kf = KFold(n_splits=3, shuffle=False)
for tr_idx, te_idx in kf.split(wine):
    print(te_idx)

# test 데이터 기준으로 확인해보면 될 것 같음
kf = KFold(n_splits=3, shuffle=True, random_state=42)
results = []
for tr_idx, te_idx in kf.split(wine):
    print(te_idx)
    model = LogisticRegression()
    train = wine.loc[tr_idx,:]
    test = wine.loc[te_idx,:]
    model.fit(train.drop("class",axis=1), train["class"])
    pred = model.predict(test.drop("class",axis = 1))
    res = accuracy_score(test["class"], pred)
    results.append(res)
np.mean(results) # 정확도: 0.7795 

ks = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for tr_idx, te_idx in ks.split(wine, wine["class"]):
    train = wine.loc[tr_idx,:]
    test = wine.loc[te_idx,:]
    print("tr_class:", train["class"].value_counts())
    print("te_class:", test["class"].value_counts())

kf = KFold(n_splits=3, shuffle=False)
for i, (tr_idx, te_idx) in enumerate(kf.split(range(9))):
    print(f"{i}폴드 tr_idx:", tr_idx)
    print(f"{i}폴드 te_idx:", te_idx)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (tr_idx, te_idx) in enumerate(kf.split(range(9))):
    print(f"{i}폴드 tr_idx:", tr_idx)
    print(f"{i}폴드 te_idx:", te_idx)


#################
# 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
#GridSearchCV(model, 옵션, cpu 개수)

model = DecisionTreeClassifier(random_state=42)
params = {"min_impurity_decrease":np.arange(0.0001, 0.0006, 0.0001)}
gs = GridSearchCV(model, params, n_jobs=-4)
dir(gs)
train, test = train_test_split(wine, test_size = 0.1, random_state=42)
gs.fit(train.drop("class", axis =1), train["class"])
gs.n_splits_
model = gs.best_estimator_

pred = model.predict(test.drop("class",axis = 1))
accuracy_score(test["class"], pred)

################ 
# kfold 5개가 아니라 다르게 주고 싶을 때 
kf = KFold(n_splits=10,random_state=42, shuffle=True)
# scoring 옵션 통해서 함수 설정 가능
gs = GridSearchCV(model, params, n_jobs=-4, cv = kf) 
train, test = train_test_split(wine, test_size = 0.1)
gs.fit(train.drop("class", axis =1), train["class"])
gs.n_splits_
gs.best_params_ # 최종 선택된 파라미터 값
gs.best_estimator_ # 최종 선택된 모델
gs.best_score_ # 최종 모델의 스코어 

model = gs.best_estimator_
pred = model.predict(test.drop("class",axis = 1))
accuracy_score(test["class"], pred)

params = {
    "min_impurity_decrease":np.arange(0.0001, 0.0006, 0.0001),
    "max_depth": range(5, 20, 1),
    "min_samples_split":range(2, 100, 10)
    }
train, test = train_test_split(wine, test_size = 0.1)
model = DecisionTreeClassifier(random_state=42)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
gs = GridSearchCV(model, params, n_jobs=-4, cv = kf, verbose=1) 
gs.fit(train.drop("class", axis =1), train["class"])
model = gs.best_estimator_
pred = model.predict(test.drop("class",axis = 1))
accuracy_score(test["class"], pred)

####
from scipy.stats import uniform, randint
params = {
    "min_impurity_decrease":uniform(0.0001, 0.001),
    "max_depth": randint(20,50),
    "min_samples_split":randint(2,25),
    "min_samples_leaf":randint(1,25)
    }

train, test = train_test_split(wine, test_size = 0.1)
model = DecisionTreeClassifier(random_state=42)

rs =  RandomizedSearchCV(
    model, params, random_state=42, n_jobs=-4, n_iter=100, verbose=1)
rs.fit(train.drop("class", axis =1), train["class"])
rs.best_params_
model = rs.best_estimator_
pred = model.predict(test.drop("class",axis = 1))
accuracy_score(test["class"], pred)

########################
# https://federated-xgboost.readthedocs.io/en/latest/python/python_api.html

# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
import seaborn as sns
df = sns.load_dataset("mpg")
df = df[~ df.isna().any(axis=1)].reset_index(drop=True)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
train, test = train_test_split(df, test_size=0.1, random_state=42)

# mpg 예측 성능 MAE로 평가
test["mpg"] 

############################45분에 풀이
tr_X = train.drop(["mpg","acceleration", "name"],axis = 1)
tr_X["origin"] = tr_X["origin"].map({"usa":0,"japan":1,"europe":2})
te_X = test.drop(["mpg","acceleration", "name"],axis = 1)
te_X["origin"] = te_X["origin"].map({"usa":0,"japan":1,"europe":2})
tr_y = train['mpg']

xgb_parmas = {
    "n_estimators":range(100,1100, 100),
    "max_depth":range(3, 11, 2),
    "learning_rate":np.arange(0.01, 0.05, 0.01),
    #"reg_alpha":np.arange(0.0,1.0,0.1),
    #"reg_lambda":range(1,10, 2)
    }
model = XGBRegressor(random_state=42)
gs = GridSearchCV(model, xgb_parmas, n_jobs=-1, 
                  scoring="neg_mean_squared_log_error", verbose=1)
gs.fit(tr_X, tr_y)
model = gs.best_estimator_
pred = model.predict(te_X)
mean_absolute_error(test['mpg'], pred) # 1.7285

###########
#pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf_bs_params = {
    "n_estimators":Integer(50, 500),
    "max_depth":Integer(3, 20),
    "min_samples_split":Integer(2, 10),
    "min_samples_leaf":Integer(1, 5)
    }
model = RandomForestRegressor(random_state=42)
bs=BayesSearchCV(model, rf_bs_params, n_iter=30, n_jobs=-1, 
                 scoring="neg_mean_squared_log_error", verbose=1)
bs.fit(tr_X, tr_y)
model = bs.best_estimator_
pred = model.predict(te_X)
mean_absolute_error(test['mpg'], pred) # 1.6298

bs.best_params_
final_model = RandomForestRegressor(**bs.best_params_, random_state=42)
#RandomForestRegressor(max_depth=9, min_samples_leaf=3, min_samples_split=4, n_estimators=69)
final_model.fit(tr_X, tr_y)
pred = model.predict(te_X)
mean_absolute_error(test['mpg'], pred) # 1.728
final_model.get_params() # 파라미터 확인

