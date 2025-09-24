"""
- ID : 샘플별 고유 ID
- age : 고객 나이 ( 18 ~ 65)
- gender : 고객 성별 ( M(56%), F(44%) )
- tenure : 고객이 서비스를 이용한 총 기간 (월)  1~60 개월
- frequent : 고객의 서비스 이용일 (1~30 일)
- payment_interval : 고객의 결제 지연일 (1~30 일)
- subscription_type : 고객의 서비스 등급 ( member, plus, vip)
- contract_length : 고객의 서비스 계약 기간 (30, 90, 360)
- after_interaction : 고객이 최근 서비스 이용으로부터 경과한 기간 (일)
- support_needs : 고객의 지원 필요도 (0 : 낮음 , 1 : 중간 , 2 : 높음)
"""
import os 
import pandas as pd 
import numpy as np 
os.chdir("E:/cjcho_work/250924")

train = pd.read_csv("open/train.csv")
test = pd.read_csv("open/test.csv")
# 연령대
# frequent/tenure  # 서비스 이용 밀도
# payment_interval>np.mean(payment_interval) # 평균 이상으로 지연되는지의 유무
# after_interaction < 7 최근 7 이내에 이용했는지 대한 유무
# payment_interval / contract_length 결제 지연일 / 계약 기간 

# 2시 12분
# train test 에 대해서 위 파생 변수 만들어보기
train["age_group"] = pd.cut(train["age"], bins=range(10,81,10),right=False)
test["age_group"] = pd.cut(test["age"], bins=range(10,81,10),right=False)

# 서비스 밀도
train["service_density"] = train["frequent"]/train["tenure"]
train["is_delay"] = (train["payment_interval"] > train["payment_interval"].mean()
                     ).astype(int)
# 최근 7일 이내 사용 유무
train["is_recently_used"] = (train["after_interaction"]<7).astype(int)

# 지연 강도
train["delay_payment"] = train["payment_interval"]/train["contract_length"]

# 서비스 밀도
test["service_density"] = test["frequent"]/test["tenure"]
test["is_delay"] = (test["payment_interval"] > test["payment_interval"].mean()
                     ).astype(int)
# 최근 7일 이내 사용 유무
test["is_recently_used"] = (test["after_interaction"]<7).astype(int)

# 지연 강도
test["delay_payment"] = test["payment_interval"]/test["contract_length"]

### 
# ID gender subscription_type
train["subscription_type"]
train.info()
train[["member","plus","vip"]] = pd.get_dummies(train["subscription_type"]).astype(int)
train[["gender_F","gender_M"]] = pd.get_dummies(train["gender"]).astype(int)

test[["member","plus","vip"]] = pd.get_dummies(test["subscription_type"]).astype(int)
test[["gender_F","gender_M"]] = pd.get_dummies(test["gender"]).astype(int)

train = train.drop(["gender", "subscription_type"], axis = 1)
test = test.drop(["gender", "subscription_type"], axis = 1)


col=['[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '[60, 70)']
pd.get_dummies(train["age_group"].astype(str)).astype(int).columns
train[col] = pd.get_dummies(train["age_group"].astype(str)).astype(int)
test[col] = pd.get_dummies(test["age_group"].astype(str)).astype(int)

train = train.drop("age_group", axis = 1)
test = test.drop("age_group", axis = 1)
## 26분 시작
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 종속변수 support_needs
train["support_needs"].value_counts()

# 33분까지 모델 3개 만들고 학습하기.
rf = RandomForestClassifier(random_state=42)
# rf.fit(X, y)
rf.fit(train.drop(["ID","support_needs"], axis=1),train["support_needs"])

######
# tree 기반 모델은 독립변수의 더미화가 반드시 필수는 아님(contract_length)
lgb = LGBMClassifier(random_state=42)
lgb.fit(
        # 컬럼명에 문제가 발생하면 values 로 numpy 형태로 변환하면 해결됨
        train.drop(["ID","support_needs"], axis=1).values,
        train["support_needs"])
xgb = XGBClassifier(random_state = 42)
xgb.fit(
        # 컬럼명에 문제가 발생하면 values 로 numpy 형태로 변환하면 해결됨
        train.drop(["ID","support_needs"], axis=1).values,
        train["support_needs"])

pred = rf.predict(test.drop(["ID"], axis=1))
pd.Series(pred).value_counts()
train["support_needs"].value_counts()

submission = pd.read_csv("open/sample_submission.csv")
submission["support_needs"] = pred
submission.to_csv("00_rf.csv", index = False)

## 자료가 불균형 할 때, 균형적으로 학습하기 위해서 특정 레이블에 대해 
# 가중치를 부여하는 기술

weights = {0:1, 1:5, 2:10}
rf = RandomForestClassifier(random_state=42, class_weight=weights)
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
# rf.fit(X, y)
rf.fit(train.drop(["ID","support_needs"], axis=1),train["support_needs"])
train["support_needs"].value_counts()

##### step 1 : 지원이 필요(1,2)한지 안필요(0)한지 유무
rf_step1 = RandomForestClassifier(random_state=42)
target = train["support_needs"].isin([1,2]).astype(int)
rf_step1.fit(train.drop(["ID","support_needs"], axis=1),target)

##### step 2 : 1인지 2인지 
temp = train.copy()
train["support_needs"].isin([1,2])
# 지원이 필요한 데이터만 추출
temp = temp[ train["support_needs"].isin([1,2])]
target = temp["support_needs"].isin([2]).astype(int)
rf_step2 = RandomForestClassifier(random_state=42)
rf_step2.fit(temp.drop(["ID","support_needs"], axis=1),target)

### 예측 30분까지 수행 
pred = rf_step1.predict(test.drop(["ID"],axis=1))
pred==1
# 지원이 필요한 데이터만 추출
temp = test[pred==1]

# 추출된 데이터로 예측
pred2 = rf_step2.predict(temp.drop(["ID"],axis=1)) + 1

# 예측된 데이터를 첫번째 예측에 반영
pred[pred==1] = pred2


pd.Series(pred).value_counts()
submission = pd.read_csv("open/sample_submission.csv")
submission["support_needs"] = pred
submission.to_csv("01_rf_two_step.csv", index = False)
####################################
#### 하이퍼 파라미터 써치는 하되 너무 시간 투자하지말고 파생변수를 추가하는 형식으로
lgb = LGBMClassifier(random_state=42)
lgb_params = {
    "n_estimators":[100, 200, 500, 1000],
    "max_depth": [-1, 5, 10, 20],
    "num_leaves":[30, 60, 120],
    "learning_rate": [ 0.01, 0.05, 0.1],
    "subsample":[0.6, 0.8, 1.0],
    "colsample_bytree":[0.6, 0.8, 1.0]
    }

from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgb, # 모델
    param_distributions=lgb_params, # 서치할 파라미터 범위
    n_iter = 10, # 랜덤하게 10번 옵션 바꿔서
    cv=5, # 학습데이터를 5등분해서 평가
    scoring="f1_weighted", # 평가 기준은 f1 score를 class weight 줘서
    random_state = 42
    )

random_search.fit(
        # 컬럼명에 문제가 발생하면 values 로 numpy 형태로 변환하면 해결됨
        train.drop(["ID","support_needs"], axis=1).values,
        train["support_needs"])


lgb = LGBMClassifier(random_state=42)
lgb.num_leaves

lgb = LGBMClassifier(**random_search.best_params_, random_state=42)
lgb.fit(
        # 컬럼명에 문제가 발생하면 values 로 numpy 형태로 변환하면 해결됨
        train.drop(["ID","support_needs"], axis=1).values,
        train["support_needs"])

pred = lgb.predict(test.drop(["ID"], axis=1))
pd.Series(pred).value_counts()

submission = pd.read_csv("open/sample_submission.csv")
submission["support_needs"] = pred
submission.to_csv("02_lgb_search_random.csv", index = False)


#### 하이퍼파라미터 튜닝 하면서 투스텝으로 진행하기
