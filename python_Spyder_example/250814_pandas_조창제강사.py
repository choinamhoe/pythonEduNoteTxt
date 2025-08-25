# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 14:18:32 2025

@author: human
"""

# pip install pandas 
import pandas as pd

obj = pd.Series([4, 7, -5, 3])
obj.array
obj.values
obj.index
# 인덱스는 딕셔너리에 Key 값에 해당
obj2 = pd.Series([4,7,-5,3], index = ["d","b","a","c"])


obj2[2]
obj2["a"]

obj
obj_dict = {0:4, 1:7, 2:-5, 3:3}
obj_dict[2]
pd.Series(obj_dict) # 딕셔너리 형태와 동일

obj2[["b","a"]]
obj2[["b","a"]] = 1
obj2

obj2 = pd.Series([4,7,-5,3], index = ["d","b","a","c"])
obj2[obj2>0]
obj2[obj2>0] * 2
obj2[obj2>0] **2

import numpy as np 
np.exp(obj2[obj2>0])

obj2.to_dict()
# 일부러 결측 데이터 발생
sdata = {"Ohio":35000, "Texas":71000,
         "Oregon":16000,"Utah":5000}
states = ["Califonia","Ohio","Oregon","Texas"]
obj4 = pd.Series(sdata, index= states)
obj4.isna() # 관측값이 결측인지 True/False로 반환
obj4.isna().sum() # 결측값의 총 개수

# 결측치가 아니면 이라고 not 을 주고싶으면 ~ 활용
(~obj4.isna()) 

obj5=obj4.copy()
obj5["Califonia"] = 50000
obj5
obj4
obj4 + obj5

# 이름 표기(안씀)
obj4.name = "population"
obj4.index.name = "state"

data = {
        "state":["Ohio"]*3 + ["Nevada"]*3,
        "year":[2000,2001,2002,2001,2002,2003],
        "pop":[1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
        }
frame = pd.DataFrame(data)
frame.head()
frame.tail(5)
pd.DataFrame(data, columns = ["year","state","pop"])
frame2 = pd.DataFrame(
    data, 
    columns = ["year","state","pop","debt"])

frame2.columns
frame2.columns = ["Year","State","Pop","Debt"]
frame2.columns
frame2["Debt"]
frame2.iloc[4:6,:]
frame2.iloc[4:6,1:3]
frame2.loc[4:6,["State","Pop"]]

frame3 = frame2[1:4].copy()
# 0, 1, 2 라는 인덱스 중 0은 존재하지 않음
# 원래는 오류를 반환
frame3.loc[0:2,:] 
frame3.reset_index(drop=True) # 인덱스 초기화

# 컬럼을 생성/삭제
frame2["Debt"] = 16.5
frame2["Debt"] = np.arange(6)
frame2["Debt"] = range(6)
frame2["Debt"] = [1,2,3,4,5,6]
# 없는 컬럼에 특정 값을 넣어서도 컬럼 생성 가능
frame2["test"] = 0 
frame2["Eastern"] = frame2["State"]=="Ohio"
frame2 = frame2.drop("test",axis=1)

# test라는 컬럼을 생성하고
# 연도가 2001, 2002년도인 경우에 대해서 True
# 아닌 경우를 False로 생성 
frame2.loc[:,"test"] = False
frame2["test"] = False
cond = frame2["Year"].isin([2001,2002])
frame2.loc[cond, "test"]=True

frame2["test"] = False
frame2.loc[frame2["Year"]==2001,"test"]=True
frame2.loc[frame2["Year"]==2002,"test"]=True

populations = {
    "Ohio":{2000:1.5, 2001:1.7, 2002:3.6},
    "Nevada":{2001:2.4, 2002:2.9}
    }
frame3 = pd.DataFrame(populations)
frame3.T
# 자료의 타입이 자동으로 변환됨
populations = {
    "Ohio":{2000:"aaa", 2001:"aaa", 2002:"3.6"},
    "Nevada":{2001:2.4, 2002:2.9}
    }
frame3 = pd.DataFrame(populations)
frame3.T # 전치 (행과 열을 바꿈)
frame3.dtypes # Ohio는 문자열, Nevada 는 실수
frame3.T.dtypes # 2000, 2001, 2002 모두 문자열로 출력


populations = {
    "Ohio":{2000:1.5, 2001:1.7, 2002:3.6},
    "Nevada":{2001:2.4, 2002:2.9}
    }
frame3 = pd.DataFrame(populations)

frame3.to_numpy()
frame3.to_dict()

populations = {
    "Ohio":{2000:"aaa", 2001:"aaa", 2002:"3.6"},
    "Nevada":{2001:2.4, 2002:2.9}
    }
frame3 = pd.DataFrame(populations)
# 넘파이 객체는 자료타입을 1개 밖에 가지지 못함
frame3.to_numpy() 

data = pd.DataFrame(
    np.arange(16).reshape(4,4),
    index=["Ohio","Colorado","Utah","New York"],
    columns = ["One","Two","Three","Four"]
    )
# One Three 이름의 열 제거
data.drop(["One","Three"],axis=1) 
# Colorado 이름의 행 제거
data.drop(["Colorado"], axis=0)

data.drop(["One","Three"], axis="columns")
data.drop(["Colorado"], axis="rows")

# New York 행과 Two 열을 제거하시오
# 인덱스 값을 컬럼으로 만들어보시오
## Type1
data.drop("New York", axis=0).drop("Two", axis = 1)
## type2
data2=data.drop("New York", axis=0)
data2.drop("Two", axis = 1)

data2=data.drop("New York", axis=0).drop("Two", axis = 1)
data2.reset_index()

# One 컬럼을 제외한 테이블을 만들고, 컬럼별 평균을 구하시오
# drop 함수를 활용한 방식
data.drop("One",axis = 1).mean(axis=0)
# 변수 선택을 활용한 방식
data.loc[:,["Two","Three","Four"]]
# 차집합을 활용한 방식
selected_columns = set(data.columns) - set(["One"])
data.loc[:,list(selected_columns)]
