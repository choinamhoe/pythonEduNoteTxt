# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 12:39:49 2025

@author: human
"""

import pandas as pd
populations = {
    "Ohio":{2000:1.5, 2001:1.7, 2002:3.6},
    "Nevada":{2001:2.4, 2002:2.9}
    }
frame3 = pd.DataFrame(populations)
frame3

import numpy as np
np.nan
#결측치 개수 새보기
# 열별 결측치 개수
missing_per_column = frame3.isna().sum()
print("열별 결측치 개수:")
print(missing_per_column)

# 전체 결측치 개수
total_missing = frame3.isna().sum().sum()
print("\n전체 결측치 개수:")
print(total_missing)

#하나라도 결측인 행은 모두 결측으로 만들기
## type1
cond = (frame3["Ohio"].isna()) | (frame3["Nevada"].isna())
frame3.loc[cond,:]= np.nan
frame3
## type2
frame3[frame3.isna().any(axis=1)] = None
frame3[frame3.isna().any(axis=1)] = np.nan
frame3

#Navada 행이 결측인 경우 -999로 변경하기
frame3.loc[frame3["Nevada"].isna(), "Nevada"] = -999
frame3

"""
page 209
"""

data = pd.DataFrame(
        np.arange(16).reshape((4,4)),
        index = ["Ohio","Colorado","Utah","New York"],
        columns = ["one","two","three","four"]
    )
data
data["two"] # data.loc[:,"two"]
data[["three","one"]] #data.loc[:,["three","one"]]
data[:2] #data.iloc[:2,:]
data[data["three"]>5] #data.iloc[data.loc[:,"three"]>5,:]

ser1 = pd.Series(np.arange(3.0))
ser1[-1] # ser1.loc[-1]
ser1.iloc[-1]
ser2 = pd.Series(np.arange(3.0), index=["a","b","c"])
ser2
ser2[-1]

"""
page 216
"""
data = pd.DataFrame({
    "one":[1,1,1,1],
    "two":[0,5,9,13],
    "three":[0,6,10,14],
    "four":[0,7,11,15],
    },
    index=["Ohio","Colorado","Utah","New York"]
    )
data.iloc[2] = 5 # data.iloc[2,:]
data
# data.loc[data.loc[:,"four"]>5, :] = 3
data.loc[data["four"]>5] = 3 
# data.loc[data.three==5,"three"] = 6
data.loc[data.three==5]["three"] = 6
data
data2 = data.loc[data.three==5]
data2["three"] = 6
data2

"""
page 219
"""
df1 = pd.DataFrame(
    np.arange(9).reshape((3,3)), 
    columns = list("bcd"),
    index = ["Ohio", "Texas","Colorado"]
    )
df1
df2 = pd.DataFrame(
    np.arange(12).reshape((4,3)), 
    columns = list("bde"),
    index = ["Utah", "Ohio", "Texas","Oregon"]
    )
df2
df1 + df2
"""
키값, 컬럼 값으로 정렬
"""
df1 = pd.DataFrame(
    np.arange(9).reshape((3,3)), 
    columns = list("bcd"),
    index = list("cad")
    )
df1
df1.iloc[1,0]=9
df1
df1.sort_index() # 키값 기준으로 정렬
df1.sort_values("b") # 아무것도 소트가 안적혀 있을 경우 asc 오름차순
df1.sort_values("b",ascending=True) # 특정 컬럼 기준으로 정렬
df1.sort_values("b",ascending=False)
"""
중복색인
"""
obj = pd.Series(np.arange(5),index=list("aabbc"))
obj
obj.loc["a"]
obj.loc["c"]

"""
기술통계량
상관계수 : 두 변수 간에 선형적인 상관 관계를 1~-1로 표현한 값
공분산 : 두 변수 간에 선형적인 관계를 표현한 값이지만
단위를 반영하므로 단위가 다르면 직접적인 비교가 어려움
"""
df1 = pd.DataFrame(
    np.arange(9).reshape((3,3)),
    columns = list("bcd"),
    index = list("cad")
    )
df1
df1.describe()
df1["c"].corr(df1["d"])

"""
카테고리 개수 파악
"""
obj = pd.Series(np.arange(5),index=list("aabbc"))
obj
obj = obj.reset_index()
obj
obj["index"].value_counts()

file_path = "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/META_관측지점정보_20250814152343.csv"
df = pd.read_csv(file_path,encoding='cp949')
df

#지점별 관측치(행) 개수 새기
df.loc[:,"지점"].value_counts()
df.columns
df.loc[df["지점"]==155,['지점', '시작일', '종료일','위도', '경도', '노장해발고도(m)']]
#지점별 마지막 값 추출하기
df.drop_duplicates("지점",keep="last")
#지점별 첫 값 추출하기
df.drop_duplicates("지점",keep="first")
#컬럼별 결측치 개수 새기
df.isna().sum()

file_path2 = "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/수원7월데이터.csv"
obs_df1 = pd.read_csv(file_path2,encoding='cp949')
obs_df1
file_path3 = "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/수원8월데이터.csv"
obs_df2 = pd.read_csv(file_path3,encoding='cp949')
obs_df2
obs_df1.head(3)
obs_df2.head(3)
#지점, 일시, 강수량(mm)
obs_df1 = obs_df1.loc[:,['지점','일시','강수량(mm)']]
obs_df2 = obs_df2.loc[:,['지점','일시','강수량(mm)']]
obs_df1.columns
obs_df2.columns
obs_df1
obs_df2
obs_df1.head(3)
obs_df2.head(3)
obs_df1.shape
obs_df2.shape
pd.concat([obs_df1,obs_df2],axis=0)
pd.concat([obs_df1,obs_df2],axis=1)
####

file_path = "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/META_관측지점정보_20250814152343.csv"
location_df = pd.read_csv(file_path,encoding='cp949')
selected_columns = [
    "지점","시작일","종료일",
    "위도","경도","노장해발고도(m)"]
location_df = location_df.loc[:,selected_columns]
location_df
location_df = location_df.drop_duplicates(
    "지점", keep="first")
location_df
location_df = location_df.reset_index(drop=True)
location_df

obs_df = pd.concat([obs_df1,obs_df2],axis=0,ignore_index=True)
obs_df
obs_df_merge_left = pd.merge(obs_df, location_df, on="지점", how="left")
obs_df_merge_left
obs_df_merge_right = pd.merge(obs_df, location_df, on="지점", how="right")
obs_df_merge_right
