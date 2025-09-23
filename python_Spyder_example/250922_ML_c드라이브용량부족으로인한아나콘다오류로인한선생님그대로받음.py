"""
# 항공사 만족도 예측
https://dacon.io/competitions/official/235871/overview/description

# 데이터 다운로드(안들어가도 무방)
https://www.kaggle.com/code/neginfirouzi/airline-passenger-satisfaction-predictive-analysis

# 분할 해놓은 데이터
http://ducj.iptime.org:5000/fsdownload/LE8gqnelk/air

# 8월 28일자 머신러닝 실습 코드 참고하면 좋을 듯

1~ 300줄 rf, xgb, lgb 모델 만들고 성능 비교
300~ 줄 3개 모델을 합쳐서 stacking 모델 생성 후 성능 비교
"""
dir = "E:/최남회/파이썬개발에대한파일모음/air"
import os
import pandas as pd 
os.chdir(dir)

train = pd.read_csv("train.csv")
# 컬럼 타입 확인
train.info()
# 기초통계량 확인
des = train.describe()
# 결측치 확인
train.isna().sum() 

test = pd.read_csv("test.csv")

import matplotlib.pyplot as plt 
# pip install seaborn
import seaborn as sns
%matplotlib auto

corr_mat=train.drop(
    ["id","Gender", "Customer Type",
     "Type of Travel", "Class"],axis=1).corr()

# 인사이트 도출 및 파생변수 도출에 활용, 교호작용이 있으면 둘 중 하나의 컬럼만 활용
# Departure Delay in Minutes : 출발 지연 시간
# Arrival Delay in Minutes : 도착 지연 시간
# 늦게 출발하면 당연히 늦게 도착해서 양의 상관관계를 가짐 
# 회귀 모델들에서는 독립변수 사이에 상관성이 높으면 오히려 예측성능에 악영향을 끼치는
# 교호작용 현상이 나타남. 
# 일반적으로 두 컬럼의 곱을 변수로 추가하거나 하나의 컬럼을 제거하는 형태로 해소

sns.heatmap(corr_mat, cmap="Blues", fmt=".2f", annot=True)

plt.imshow(corr_mat, cmap='Blues')
plt.xticks(
    ticks=range(len(corr_mat.columns)), 
    labels=corr_mat.columns, rotation=90)
plt.yticks(
    ticks=range(len(corr_mat.columns)), 
    labels=corr_mat.columns)
plt.colorbar()
for i in range(len(corr_mat)):
    for j in range(len(corr_mat)):
        plt.text(j, i, f"{corr_mat.iloc[i, j]:.2f}", 
                 ha='center', va='center', color='black')


# 남자 여자 비율(여자 > 남자)
sns.countplot(x="Gender", data = train)
sns.countplot(x="Gender", data = test)

plot_df = train["Gender"].value_counts()
plt.bar(plot_df.index, plot_df.values)

# 학습 데이터의 성별 분포 확인
plt.hist(train["Age"], bins=10,density=True, alpha=0.5)
plt.hist(test["Age"], bins=10,density=True, alpha=0.5)

# bar plot 시각화 
# 남성이 여성 대비 불만족도가 높음
sns.countplot(x="Gender", hue = "target", data=train)

# 여행에 목적에 따른 만족도
sns.countplot(x="Type of Travel", hue = "target", data=train)
# 클레스와 여행에 목적에 따른 만족도
# 업무적으로 갔을 때 비즈니스 빈도가 많더라
sns.countplot(x="Class", hue = "Type of Travel", data=train)

# 업무적으로 갔을 때 비즈니스를 타게 되면 만족도가 높더라
train["Type of Travel"].value_counts()
extract = train[train["Type of Travel"]=="Business travel"]
sns.countplot(x="Class", hue = "target", data=extract)

# 거리에 따른 만족도
train.columns
plt.hist(train["Flight Distance"], bins=10,density=True, alpha=0.5)
plt.hist(test["Flight Distance"], bins=10,density=True, alpha=0.5)

sns.boxplot(x="target", y= "Flight Distance", data=train)
sns.boxplot(x="target", y= "Flight Distance", hue="Class", data=train)

# 거리에 따른 만족도
# 1200~ 2300 불만족도가 높아짐(중거리)
sns.countplot(x=train["Flight Distance"]//100, hue = "target", data=train)

extract = train[(train["Flight Distance"]>=1200)&(train["Flight Distance"]<=2300)]

corr_mat_v2=extract.drop(
    ["id","Gender", "Customer Type",
     "Type of Travel", "Class"],axis=1).corr()
# 전체 데이터의 만족도 대비 1200~2300 데이터의 만족도의 상관계수의 변화량
### 수하물 처리 만족도, 식음료 만족도 등의 만족도가 높아야 고객만족도가 올라감
### 중거리 고객들은 다른 고객 대비 서비스의 질이 중요하다.
sns.heatmap((corr_mat-corr_mat_v2).abs(), annot=True, cmap = "Blues", fmt = ".2f")

# 서비스 만족도
"""
좌석 만족도
식음료 만족도
게이트 위치 만족도
와이파이 만족도
"""
# 좌석 만족도가 1~3이어도 만족하는 사람이 있었다.
sns.countplot(x="Seat comfort", hue="target", data=train)
# 식음료 만족도도 1~3이어도 만족하는 사람이 있었다.
sns.countplot(x="Food and drink", hue="target", data=train)
# 다른 만족도 대비 차이가 크지 않음
sns.countplot(x="Gate location", hue="target", data=train)
# 와이파이 만족도  엄청 느리면 불만족 하는 것 같고 적정 속도가 나오면 만족하는 것 같기도 함
sns.countplot(x="Inflight wifi service", hue="target", data=train)

# 생각
# 1. 다른 요인이 있어서 만족도가 낮아도 고객 만족하는 경우가 있다고 생각이 듬
# 2. 국내는 0 혹은 5에 만족도를 선택하는 경향이 강한데 해외는 1~3도 많이들 선택
## 12분 시작

# 지연시간에 따른 만족도
#plt.plot(train["Departure Delay in Minutes"])
train["Departure Delay in Minutes"].describe()
# 출발 지연 시간이 100을 초과하게 되면 불만족도들이 높아진다.
extract = train[train["Departure Delay in Minutes"]>100]
sns.countplot(x=extract["Departure Delay in Minutes"]//10, hue = extract["target"])

# 출발 지연 시간이 100이하에서는 큰 차이가 없다.
extract_v2 = train[train["Departure Delay in Minutes"]<=100]
sns.countplot(x=extract_v2["Departure Delay in Minutes"]//10, hue = extract_v2["target"])

## 26분 시작
train.info()
import sklearn
sklearn.__version__
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500, random_state=42)

# target 변수를 제외하고 모두 다 동일
set(train.columns) - set(test.columns)

train["Gender"] = train["Gender"].map({"Female":0, "Male":1})
train.info()
train["Customer Type"] = train["Customer Type"].map(
    {"disloyal Customer":0, "Loyal Customer":1})

train["Type of Travel"] = train["Type of Travel"].map(
    {"Personal Travel":0, "Business travel":1})

class_dummies = pd.get_dummies(train["Class"])+0

train_df = train.drop(["id", "Class"], axis = 1)
train_df = pd.concat([train_df,class_dummies],axis=1)

##

test["Gender"] = test["Gender"].map({"Female":0, "Male":1})
test["Customer Type"] = test["Customer Type"].map(
    {"disloyal Customer":0, "Loyal Customer":1})
test["Type of Travel"] = test["Type of Travel"].map(
    {"Personal Travel":0, "Business travel":1})

class_dummies = pd.get_dummies(test["Class"])+0

test_df = test.drop(["id", "Class"], axis = 1)
test_df = pd.concat([test_df,class_dummies],axis=1)

# 컬럼 명에서 띄어쓰기나 /를 _로 변경
train_df.columns = [i.replace(" ","_").replace("/", "_") for i in train_df.columns]
test_df.columns = [i.replace(" ","_").replace("/", "_") for i in test_df.columns]

train_df.columns

train_df.shape
test_df.shape

model.fit(train_df.drop("target",axis=1), train_df["target"])
pred = model.predict(test_df)

# submit 테이블이랑 test 테이블이랑 ID 가 다를 수 있으니 join 하는 형태로 제출
result = test[["id"]].copy()
result["pred"] = pred
submit = pd.read_csv("sample_submission.csv")
submit = pd.merge(submit, result)
submit["target"] = submit["pred"]
submit = submit.drop("pred",axis=1)
submit.to_csv("first_result.csv", index=False)


true_df = pd.read_csv("true_label.csv")
pred_df = pd.read_csv('first_result.csv')
# 예측값이랑 실제값이랑 합치기
checked_df = pd.merge(true_df, pred_df, on="id")
checked_df.columns = ["id", "true", "pred"]

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(checked_df["true"], checked_df["pred"])
cm = confusion_matrix(checked_df["true"], checked_df["pred"])
pd.DataFrame(cm , index=["True", "False"], columns=["neg", "pos"])

# 15분 시작
import sklearn
sklearn.__version__

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
params = {
    "max_depth":randint(1, 20),
    "min_samples_split":randint(2,20),
    "max_features":["auto", "sqrt", "log2"]
    }

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter = 10,
    cv = 5,
    scoring="accuracy",
    random_state=42)

random_search.fit(train_df.drop("target",axis=1), train_df["target"])
# 하이퍼 파라미터 튜닝

best_params = random_search.best_params_
best_params["n_estimators"] = 500

# 하이퍼 파라미터 튜닝은 cross validation 활용 했으니 전체 데이터로 다시 학습 수행
# 딕셔너리를 함수에 **dict 로 집어 넣으면 키값은 옵션명 value 값은 값으로 입력됨
new_model = RandomForestClassifier(**best_params,random_state=42)
new_model.max_depth
new_model.max_features
new_model.min_samples_split

# 31 분 시작
new_model.fit(train_df.drop("target",axis=1), train_df["target"])
pred = new_model.predict(test_df)
checked_df["pred"] = pred
# 아래는 편의상 가져와서 수행
accuracy_score(checked_df["true"], checked_df["pred"]) # 0.919
cm = confusion_matrix(checked_df["true"], checked_df["pred"]) 
pd.DataFrame(cm , index=["True", "False"], columns=["neg", "pos"])


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint
import random
import numpy as np 
# pip install xgboost lightgbm
xgb_model = XGBClassifier(random_state=42)
xgb_params = {
    "n_estimators":randint(100,500),
    "max_depth":randint(1, 20),
    "learning_rate":np.linspace(0.001, 0.05,50),
    "subsample":np.linspace(0.06, 0.1,50),
    }
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter = 30,
    cv = 5,
    scoring="accuracy",
    random_state=42)

random_search.fit(train_df.drop("target",axis=1), train_df["target"])
xgb_model = XGBClassifier(**random_search.best_params_, random_state=42)
xgb_model.fit(train_df.drop("target",axis=1), train_df["target"])

pred = xgb_model.predict(test_df)
# 아래는 편의상 가져와서 수행
accuracy_score(checked_df["true"], pred)
cm = confusion_matrix(checked_df["true"], pred) 
pd.DataFrame(cm , index=["True", "False"], columns=["neg", "pos"])

### 18분 시작
lgb_model = LGBMClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=xgb_params,
    n_iter = 30,
    cv = 5,
    scoring="accuracy",
    random_state=42)
random_search.fit(train_df.drop("target",axis=1), train_df["target"])
lgb_model = LGBMClassifier(**random_search.best_params_, random_state=42)
lgb_model.fit(train_df.drop("target",axis=1), train_df["target"])
pred = lgb_model.predict(test_df)
# 아래는 편의상 가져와서 수행
accuracy_score(checked_df["true"], pred)
cm = confusion_matrix(checked_df["true"], pred) 
pd.DataFrame(cm , index=["True", "False"], columns=["neg", "pos"])

## 모델 비교해서 좋은 모델은 RF다 하고 종료

## 스태킹
# 모델의 예측값을 가지고 한번더 학습해서 최종 예측값을 만드는 앙상블 알고리즘
meta_model = RandomForestClassifier(random_state = 42)

new_model, lgb_model, xgb_model
rf_pred = new_model.predict_proba(train_df.drop("target",axis=1))
xgb_pred = xgb_model.predict_proba(train_df.drop("target",axis=1))
lgb_pred = lgb_model.predict_proba(train_df.drop("target",axis=1))

# 이런 데이터에서는 이런 모델의 예측값이 좀더 유의미하고
# 이런 상황에서는 이런 모델의 예측값이 더 유의미하다는 것을 metamodel이 알아서 
# 결정하게 학습하는 방식
stacked_tr_df = pd.DataFrame({"rf":rf_pred[:,1],"xgb":xgb_pred[:,1], "lgb":lgb_pred[:,1]})
meta_model.fit(stacked_tr_df, train["target"])


### 스태킹 모델 예측
rf_pred = new_model.predict_proba(test_df)
xgb_pred = xgb_model.predict_proba(test_df)
lgb_pred = lgb_model.predict_proba(test_df)
stacked_tr_df = pd.DataFrame({"rf":rf_pred[:,1],"xgb":xgb_pred[:,1], "lgb":lgb_pred[:,1]})
# 최종적 예측값
pred = meta_model.predict(stacked_tr_df)
accuracy_score(checked_df["true"], pred)
cm = confusion_matrix(checked_df["true"], pred) 
pd.DataFrame(cm , index=["True", "False"], columns=["neg", "pos"])

