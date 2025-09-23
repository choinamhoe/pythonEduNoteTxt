# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 14:58:20 2025

@author: human
"""

# RF 예측 성능 0.9195
# xgb 예측 성능  0.8825
# lgb 에측 성능 0.887
# stacking 예측 성능  0.9135

# pip install pycaret numpy==1.23 scikit-learn 


dir = "E:/최남회/파이썬개발에대한파일모음/air"
import os
import pandas as pd 
os.chdir(dir)
from pycaret.classification import *

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

setup(
      data = train.drop("id",axis = 1),
      target = "target",
      session_id = 42
      )

top11_model = compare_models(sort="Accuracy", n_select=11)

tune_model_v1 = tune_model(top11_model[0], n_iter = 10)

pred = predict_model(tune_model_v1, data = test.drop("id", axis=1))

true_df = pd.read_csv("true_label.csv")
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(true_df["target"],pred["prediction_label"])
confusion_matrix(true_df["target"],pred["prediction_label"])

### 3시 25분 시작
# 스태킹 수행 가능
stacked_model = stack_models(estimator_list=top11_model)

# 모델 예측
pred = predict_model(stacked_model, data = test.drop("id", axis=1))
true_df = pd.read_csv("true_label.csv")
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(true_df["target"],pred["prediction_label"]) # 0.938
confusion_matrix(true_df["target"],pred["prediction_label"])