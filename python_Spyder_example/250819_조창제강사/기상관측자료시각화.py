# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:05:57 2025

@author: human
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import glob
# * 정규표현식 중 하나 
# glob.glob는 특정 패턴을 만족하는 파일 목록 가져옴
# 다운로드 폴더안에 csv파일 확장자 목록 가져옴
glob.glob("C:/Users/human/Downloads/*csv")
# 다운로드 폴더 아래에 폴더 안에 있는 csv 파일 목록 가져옴
glob.glob("C:/Users/human/Downloads/**/*csv")
# 다운로드 폴더 아래에 존재하는 모든 csv 파일 목록 가져옴
glob.glob(
    "C:/Users/human/Downloads/**/*csv",
    recursive=True)

# os: 운영체제 관련 된 작업할 때 쓰는 패키지
# shutil: 파일 복사 할 때 쓰는 패키지
# re: 정규표현식 활용할 때 쓰는 패키지
import shutil, os, re, glob

files = glob.glob("C:/Users/human/Downloads/*csv")
new_files = [i for i in files if re.compile("수원").findall(i)]
os.path.basename(new_files[0])
[shutil.copy(i,os.path.basename(i))  for i in new_files ]


### type 2 일반 for문 활용한 예제
files = glob.glob("C:/Users/human/Downloads/*csv")
file = files[0]
new_files = []
for file in files:
    if re.compile("수원").findall(file):
        new_files.append(file)

        saved_path = os.path.basename(file)
        # 특정 폴더에 저장하고 싶을 때
        saved_dir = "C:/Users/human/Desktop/새 폴더"
        saved_path = f"{saved_dir}/{saved_path}"
        ####
        shutil.copy(file, saved_path)

# type 3
os.chdir("C:/Users/human/Desktop/새 폴더") # 작업 경로 변경
os.getcwd() # 현재 작업 경로
files = glob.glob("C:/Users/human/Downloads/*csv")
file = files[0]
new_files = []
for file in files:
    if re.compile("수원").findall(file):
        new_files.append(file)
        saved_path = os.path.basename(file)
        shutil.copy(file, saved_path)

files = glob.glob("./*csv")
file = files[0]
abs_path = os.path.abspath(file) # 절대 경로 
os.path.basename(abs_path) # 파일 명 
os.path.dirname(abs_path) # 디렉토리 명

"C:/Users/human/Desktop/새 폴더/수원"
os.makedirs("C:/Users/human/Desktop/새 폴더/지점/aaa", exist_ok=True)

################################
file_dir = "C:/Users/human/Downloads"
saved_based = "C:/Users/human/Desktop/새 폴더"
saved_dir = f"{saved_based}/수원"

files = glob.glob(f"{file_dir}/*csv")
os.makedirs(saved_dir, exist_ok=True)
[shutil.copy(
    i, f"{saved_dir}/{os.path.basename(i)}") for i in files 
     if re.compile("수원|영등포").findall(i)]

for file in files:
    if re.compile("수원|영등포").findall(file):
        shutil.copy(file, f"{saved_dir}/{os.path.basename(file)}")

dir(os.path)
##################################################
saved_based = "C:/Users/human/Desktop/새 폴더"
saved_dir = f"{saved_based}/수원"
files = glob.glob(f"{saved_dir}/*csv")
selected_columns = ['지점', '지점명', '일시', '기온(°C)', '강수량(mm)', 
       '풍속(m/s)', '풍향(16방위)',  '습도(%)', '증기압(hPa)', '이슬점온도(°C)', 
       '현지기압(hPa)', '해면기압(hPa)', '일조(hr)', '일사(MJ/m2)']

dfs = list()
for file in files:
    df = pd.read_csv(file, encoding="cp949")
    df = df.loc[:,selected_columns]
    dfs.append(df)
weather_df = pd.concat(dfs,axis=0, ignore_index=True)

weather_df["일시"] = pd.to_datetime(weather_df["일시"])
pd.to_datetime(weather_df["일시"])+pd.to_timedelta(1,unit="h")

import matplotlib.pyplot as plt 
%matplotlib auto 
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(weather_df["기온(°C)"])
ax1.cla()
ax1.plot(weather_df["일시"], weather_df["기온(°C)"],label="Temperature (°C)", color = "red")
ax1.legend(loc="upper left") # (upper, center, lower), (left, center, right)
#ax1.legend(bbox_to_anchor=(0,0)) # 0~ 1 사이 범위로 위치 지정
ax1.tick_params(rotation=30, labelcolor = "black")
ax1.tick_params(axis="y", labelcolor = "red", rotation = 0 )
ax1.set_ylabel("Temperature (°C)", color="red")

ax2 = ax1.twinx() # y 축을 좌우로 활용하기 위해 사용
ax2.cla()
ax2.plot(weather_df["일시"], weather_df["습도(%)"], label = "Humidity (%)", color="blue")
ax2.tick_params(axis="y", labelcolor = "blue", rotation = 0)
ax2.set_ylabel("Humidity (%)", color="blue")
ax2.yaxis.set_label_position("right")
ax2.legend(loc="upper right")
ax1.grid(False)
ax1.grid(True, axis="x")
ax1.grid(False)
ax1.grid(True, axis="y")
ax1.grid(False)
ax1.grid(True, axis="both")
ax1.set_title("2025년 수원 데이터 7, 8월 시각화", fontsize= 32, fontweight= 700)
ax1.set_xlabel("Years")
plt.tight_layout()
plt.rcParams["font.family"] = "malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

weather_df.loc[:,"일"] = weather_df["일시"].astype(str)
values= list()
for value in weather_df.loc[:,"일"]:
    date = value[:10]
    values.append(date)
weather_df.loc[:,"일"] = values

weather_df.loc[:,"일"] = [i[:10] for i in weather_df.loc[:, "일"]]

weather_df.loc[:, "일"] = weather_df["일시"].dt.date
#ax.set_xticks([0, 25, 50])
#ax.set_xticklabels(
#    ["0년\n 월", "0년\n 월","0년\n 월"],
#    rotation=30, fontsize=12)
weather_df.head()
##################
dfs = list()
for file in files:
    df = pd.read_csv(file, encoding="cp949")
    df = df.loc[:,selected_columns]
    dfs.append(df)
weather_df = pd.concat(dfs,axis=0, ignore_index=True)
weather_df["일시"] = pd.to_datetime(weather_df["일시"])
weather_df.loc[:,"일"] = weather_df["일시"].astype(str)
weather_df.loc[:,"일"] = [i[:10] for i in weather_df.loc[:, "일"]]
# 일단위 기온 강수량 습도
weather_df.loc[weather_df.loc[:,"강수량(mm)"].isna(),"강수량(mm)"]=0
day_weather_df = weather_df.groupby("일")[["기온(°C)", "강수량(mm)","습도(%)", "풍속(m/s)"]].mean().reset_index()
# 27분까지 일단위 데이터로 일, 기온 그래프 그리기
day_weather_df["일"] = pd.to_datetime(day_weather_df["일"])
"""
7~8월 기온이 낮았을 때의 원인 파악
> 비가 오거나, 바람이 불었다.
"""
fig, ax = plt.subplots(figsize = (10,5))
ax.cla()
ax.plot(day_weather_df["일"],day_weather_df["기온(°C)"], color="red")
ax2 = ax.twinx()
ax2.plot(day_weather_df["일"], 
         day_weather_df["강수량(mm)"],color="blue")
ax2.cla()
ax2.plot(day_weather_df["일"], 
         day_weather_df["풍속(m/s)"],color="blue")
# 그림 전체에 대한 타이틀
fig.suptitle("일단위 기상 자료 시각화", fontsize=20,fontweight=700)
# sub title
ax.set_title("일 vs 기온, 풍속", fontsize=16,fontweight=400)
# x 축 레이블
ax.set_xlabel("일(YYYY-MM-DD)",fontweight=700)
# y 축 레이블
ax.set_ylabel("기온 (°C)", color="red")
ax2.set_ylabel("풍속 (m/s)", color="blue")
# 2번째 y 축 레이블이 오른쪽으로 오게 설정
ax2.yaxis.set_label_position("right")

### 풍속 시각화
fig, ax = plt.subplots(figsize = (10,5))
day_weather_df = weather_df.groupby("일")[[
    "기온(°C)","강수량(mm)","습도(%)", 
    "풍속(m/s)","풍향(16방위)"]].mean().reset_index()
day_weather_df["일"] = pd.to_datetime(day_weather_df["일"])
u = day_weather_df["풍속(m/s)"] * np.cos(
    np.deg2rad(day_weather_df["풍향(16방위)"]))
v = day_weather_df["풍속(m/s)"] * np.sin(
    np.deg2rad(day_weather_df["풍향(16방위)"]))
y_values = np.zeros(day_weather_df["일"].shape)
ax.cla()
ax.quiver(
    day_weather_df["일"],
    y_values,
    u, v, 
    headwidth = 4, headlength= 3, color="green",
    #scale= 0.05
    )

######### 2개 같이 그리기
fig, (ax1, ax2) = plt.subplots(
    nrows = 2, figsize=(10, 5),
    sharex= False, 
    gridspec_kw = {"height_ratios":[2,8]}
    )
ax1.quiver(
    day_weather_df["일"],
    y_values, u, v,
    color= "green"
    )

ax2.plot(
    day_weather_df["일"],day_weather_df["기온(°C)"], color="red")
ax3 = ax2.twinx()
ax3.plot(day_weather_df["일"], 
         day_weather_df["강수량(mm)"],color="blue")
ax4 = ax2.twinx()
ax4.plot(day_weather_df["일"], 
         day_weather_df["풍속(m/s)"],color="green")
ax4.spines["right"].set_position(("axes", 1.05))


import geopandas as gpd
shp_file = "C:/Users/human/Downloads/ctprvn_20230729/ctprvn.shp"
gdf = gpd.read_file(shp_file)
gdf = gdf.set_crs("EPSG:5179", allow_override=True)
gdf = gdf.to_crs(epsg=4326)
gdf = gdf.to_crs(epsg=3857)
ax = gdf.plot()
import contextily as ctx
ctx.add_basemap(ax, source= ctx.providers.OpenStreetMap.Mapnik)
import pandas as pd 
df = pd.read_csv(
    "C:/Users/human/Desktop/새 폴더/지점정보.csv",encoding="cp949")
df.columns
df = df.loc[:,["지점", "지점명","위도", "경도", "노장해발고도(m)"]]
location_df = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(
        df.loc[:,"경도"], 
        df.loc[:, "위도"]))
location_df = location_df.set_crs("EPSG:4326", allow_override = True)
location_df = location_df.to_crs(epsg=3857)
location_df.plot(ax = ax, color = "red")
