# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:25:15 2025

@author: human
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import datetime

now = datetime.datetime.now()
date = now.strftime("%Y%m%d")

# * 정규 표현식 중 하나
# glob.glob는 특정 패턴을 만족하는 파일 목록 가져옴
#파이썬개발에대한파일모음 폴더안에 csv파일 확장자 목록 가져옴
glob.glob("E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/*csv")
#파이썬개발에대한파일모음 폴더 아래에 폴더 안에 있는 csv파일 목록 가져옴
glob.glob("E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/**/*csv")
#파이썬개발에대한파일모음 폴더 아래에 존재하는 모든 csv파일 목록 가져옴
glob.glob(
            "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/**/*csv",
          recursive=True)

files = glob.glob(
            "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/**/*csv",
          recursive=True)
# os: 운영체제 관련 된 작업할 때 쓰는 패키지
# shutil: 파일 복사할 때 쓰는 패키지
# re: 정규표현식 활용할 때 쓰는 패키지

### type1
import os
import shutil
import re
# re.compile(date) 파일명 이름이 현재날짜(data)와 동일한지 확인할때 사용
# re.compile(date).findall(i) 결과값은 True,False로 결과값 출력
new_files = [i for i in files if re.compile(date).findall(i)]
#파일명만 찾을 때 쓰는 방식
os.path.basename(new_files[0])
#조회된 모든 파일 목록을 Spyder에 있는 경로의 파일명을 동일하게 복사
[shutil.copy(i,os.path.basename(i)) for i in new_files]

### type2 일반 for문 활용한 예제
new_files = []
for file in files:
    if re.compile(date).findall(file):
        new_files.append(file)
        saved_path = os.path.basename(file)
        #특정 폴더에 저장하고 싶을 때
        saved_dir = "E:/최남회/python_Spyder_example/"
        saved_path = f"{saved_dir}/{saved_path}"
        shutil.copy(file, saved_path)

# type 3
os.chdir("E:/최남회/python_Spyder_example/"+date) # 작업 경로 변경(스파이더 작업 경로 변경)
os.getcwd() # 현재 작업 경로 조회

files = glob.glob(
            "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/**/*csv",
          recursive=True)
file = files[0]
new_files = []
for file in files:
    if re.compile(date).findall(file):
        new_files.append(file)
        saved_path = os.path.basename(file)
        shutil.copy(file, saved_path)

#파일에 관한 함수 목록
files = glob.glob(
            "E:/최남회/python_Spyder_example/**/*csv",
          recursive=True)
file = files[0]
abs_path = os.path.abspath(file)
os.path.basename(abs_path) #파일명
os.path.dirname(abs_path) #디렉토리 명

#폴더명 생성
os.makedirs(os.path.dirname(abs_path)+date,exist_ok=True)   
        

################################ 
### 실질적인 폴더 관리 및 복사할때 경로와 관련 기본 틀 로직
##읽어드릴 파일 경로
file_dir = "E:/최남회/python_Spyder_example"
#저장할 파일 경로
saved_based = "E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털"
#저장할 경로에 현재날짜를 폴더 생성 및 생성된 폴더에 파일명 추출하여 저장 경로
saved_dir = f"{saved_based}/"+date

#파일명 읽어드리는 방법.읽어드릴 경로에 모든 하위폴더에 csv파일 모두 포함 불러옴
files = glob.glob(
            f"{file_dir}/**/*csv",
          recursive=True)
#저장할 경로에 폴더 생성
os.makedirs(saved_dir, exist_ok=True)
#저장할 경로에 현재날짜꺼만 copy
#type1
[shutil.copy(
    i, f"{saved_dir}/{os.path.basename(i)}") for i in files 
     if re.compile(date).findall(i)]

#type2
for file in files:
    if re.compile(date).findall(file):
        shutil.copy(file, f"{saved_dir}/{os.path.basename(file)}")
        
dir(os.path)
dir(shutil)
dir(glob)


################################
##읽어드릴 파일 경로
file_dir = "E:/최남회/python_Spyder_example"
selected_columns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)',  '습도(%)', '증기압(hPa)', '이슬점온도(°C)', 
       '현지기압(hPa)', '해면기압(hPa)', '일조(hr)', '일사(MJ/m2)']
#파일명 읽어드리는 방법.읽어드릴 경로에 모든 하위폴더에 csv파일 모두 포함 불러옴
#files = glob.glob(
#            f"{file_dir}/**/*csv",
#          recursive=True)
files = glob.glob(
            f"{file_dir}/{date}/*csv",
          recursive=True)
files
dfs = list()
for file in files:
    if re.compile("수원").findall(file):
        df = pd.read_csv(file,encoding='cp949')
        df = df.loc[:,selected_columns]
        dfs.append(df)
weather_df = pd.concat(dfs,axis=0)
weather_df
weather_df = pd.concat(dfs,axis=0,ignore_index=True)
weather_df["일시"] = pd.to_datetime(weather_df["일시"])
weather_df
pd.to_datetime(weather_df["일시"])[0]
pd.to_datetime(weather_df["일시"])[0] + pd.to_timedelta(1,unit="h")

import matplotlib.pyplot as plt
%matplotlib auto
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(weather_df["기온(°C)"])
ax1.cla()
ax1.plot(weather_df["일시"], weather_df["기온(°C)"],label="Temperature(°C)",color="red")
ax1.legend(loc="upper left") # (upper, center, lower),(left, center,right)
#ax1.legend(bbox_to_anchor=(0.3,1)) #0~1사이 범위로 위치 지정
ax1.tick_params(rotation=30, labelcolor = "black")
ax1.tick_params(axis="y", labelcolor = "red", rotation=0)
ax1.set_ylabel("Temperature(°C)",color="red")

ax2 = ax1.twinx() #y축을 좌우로 활용하기 위해 사용
ax2.plot(weather_df["일시"], weather_df["습도(%)"],label="Humidity(%)",color="blue")
ax2.legend(loc="upper right") # (upper, center, lower),(left, center,right)
ax2.tick_params(axis="y", labelcolor = "blue", rotation=0)
ax2.set_ylabel("Humidity(%)",color="blue")
ax2.yaxis.set_label_position("right")
ax1.grid(False)
ax1.grid(True, axis="x")
ax1.grid(False)
ax1.grid(True, axis="y")
ax1.grid(False)
ax1.grid(True, axis="both")
ax1.set_title("2025년 수원 데이터 7, 8월 시각화", fontsize = 20, fontweight = 700)
ax1.set_xlabel("날짜",color="green",fontsize = 10)
plt.tight_layout()
plt.rcParams["font.family"] = "malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

weather_df.loc[:,"일"] = weather_df["일시"].astype(str)
values = list()
for value in weather_df.loc[:,"일"]:
    date = value[:10]
    values.append(date)
weather_df.loc[:,"일"] = values
weather_df.loc[:,"일"]
weather_df.loc[:,"일"] = [i[:10] for i in  weather_df.loc[:,"일"]]
weather_df.loc[:,"일"]
weather_df.loc[:,"일"] = weather_df["일시"].dt.date
weather_df.loc[:,"일"]


dfs = list()
for file in files:
    df = pd.read_csv(file, encoding="cp949")
    df = df.loc[:,selected_columns]
    dfs.append(df)
weather_df = pd.concat(dfs,axis=0, ignore_index=True)
weather_df["일시"] = pd.to_datetime(weather_df["일시"])
weather_df.loc[:,"일"] = weather_df["일시"].astype(str)
weather_df.loc[:,"일"] = [i[:10] for i in weather_df.loc[:, "일"]]
['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)',  '습도(%)', '증기압(hPa)', '이슬점온도(°C)', 
       '현지기압(hPa)', '해면기압(hPa)', '일조(hr)', '일사(MJ/m2)']
#일단위 기온 강수량 습도
weather_df.loc[weather_df.loc[:,"강수량(mm)"].isna(),"강수량(mm)"]=0
day_weather_of = weather_df.groupby("일")[['기온(°C)', '강수량(mm)','습도(%)','풍속(m/s)']].mean().reset_index()
day_weather_of
day_weather_of["일"] = pd.to_datetime(day_weather_of["일"])
#######여기서부터 그래프 그리기 작성 start
fig, ax = plt.subplots(figsize = (10,5))
ax.plot(day_weather_of["일"],day_weather_of["기온(°C)"],label="Temperature(°C)",color="red")
#y축 레이블
ax.set_ylabel("Temperature(°C)",color="red",fontsize="15")
ax.legend(loc="upper left",fontsize="10") # (upper, center, lower),(left, center,right)
ax2 = ax.twinx()
ax2.plot(day_weather_of["일"],day_weather_of["강수량(mm)"],label="precipitation(mm)",color="blue")
ax2.set_ylabel("precipitation(%)",color="blue")
ax2.cla()
ax2.plot(day_weather_of["일"],day_weather_of["풍속(m/s)"],label="windVelocity(m/s)",color="blue")
#y축 레이블
ax2.set_ylabel("windVelocity(m/s)",color="blue",fontsize="15")
ax2.legend(loc="upper right") # (upper, center, lower),(left, center,right)

#그림 전체에 대한 타이틀
fig.suptitle("2025년 7, 8월 일단위 평균 강수량,풍속 시각화", fontsize = 20, fontweight = 700)
#sub title
ax.set_title("일 vs 기온, 풍속", fontsize = 16, fontweight = 400)
#x축 레이블
ax.set_xlabel("일(YYYY-MM-DD)",color="black",fontsize = 10)
#2번째 y축 레이블이 오른쪽으로 오게 설정
ax2.yaxis.set_label_position("right")
#######여기서부터 그래프 그리기 작성 end

### 풍속시각화
fig, ax = plt.subplots(figsize = (10,5))
day_weather_of = weather_df.groupby("일")[[
    '기온(°C)', '강수량(mm)','습도(%)'
    ,'풍속(m/s)','풍향(16방위)']].mean().reset_index()
day_weather_of["일"] = pd.to_datetime(day_weather_of["일"])

u = day_weather_of["풍속(m/s)"] * np.cos(
    np.deg2rad(day_weather_of["풍향(16방위)"]))

v = day_weather_of["풍속(m/s)"] * np.sin(
    np.deg2rad(day_weather_of["풍향(16방위)"]))

day_weather_of["일"]
u
v
#임시로 y축에 0위치에 풍향을 표시하기 위한 위치 설정
#일자별로 풍향체크
y_values = np.zeros(day_weather_of["일"].shape)
ax.quiver(
    day_weather_of["일"]
    ,y_values
    ,u
    ,v
    ,headwidth = 2
    ,headlength=3
    ,color="green"
    )

####### 2개 같이 그리기
fig, (ax1,ax2) = plt.subplots(
    nrows = 2,figsize = (10,5),
    sharex= False,
    gridspec_kw = {"height_ratios":[2,8]}
    )
ax1.quiver(
    day_weather_of["일"]
    ,y_values
    ,u
    ,v
    ,color="green"
    )
ax2.plot(day_weather_of["일"],day_weather_of["기온(°C)"],label="Temperature(°C)",color="red")
#y축 레이블
ax2.set_ylabel("Temperature(°C)",color="red",fontsize="15")
ax2.legend(loc="upper left",fontsize="10") # (upper, center, lower),(left, center,right)
ax3 = ax2.twinx()
ax3.plot(day_weather_of["일"],day_weather_of["강수량(mm)"],label="precipitation(mm)",color="blue")
ax3.set_ylabel("precipitation(%)",color="blue",fontsize="15",labelpad=0)
ax3.legend(loc="upper center") # (upper, center, lower),(left, center,right)
ax3.set_yticks([],[])  # y 축 틱 & 레이블 삭제
ax4 = ax2.twinx()
ax4.plot(day_weather_of["일"],day_weather_of["풍속(m/s)"],label="windVelocity(m/s)",color="green")
ax4.set_yticks([],[])  # y 축 틱 & 레이블 삭제
#y축 레이블
ax4.set_ylabel("windVelocity(m/s)",color="green",fontsize="15",labelpad=20)
ax4.legend(loc="upper right") # (upper, center, lower),(left, center,right)

#그림 전체에 대한 타이틀
fig.suptitle("2025년 7, 8월 일단위 평균 강수량,풍속 시각화", fontsize = 20, fontweight = 700)
#sub title
ax2.set_title("일 vs 기온, 풍속", fontsize = 16, fontweight = 400)
#x축 레이블
ax2.set_xlabel("일(YYYY-MM-DD)",color="black",fontsize = 10)
#2번째 y축 레이블이 오른쪽으로 오게 설정
ax4.yaxis.set_label_position("right")