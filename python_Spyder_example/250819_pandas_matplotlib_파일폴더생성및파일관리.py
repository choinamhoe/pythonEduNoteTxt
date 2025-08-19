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
