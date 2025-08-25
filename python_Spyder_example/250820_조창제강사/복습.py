# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:13:39 2025

@author: human
"""

"""
re: 정규표현식(Regular Expression) 관련 패키지
shutil: 파일이나 폴더 전체를 복사 해올 때 사용하는 패키지
os: 운영체제 관련된 명령어들을 사용할 수 있는 패키지
  - 폴더 경로를 가져올 때 활용하거나
  - 작업 경로를 가져올 때 활용
glob: 정규표현식을 바탕으로 특정 파일 가져올 때 사용하는 패키지
"""
import glob, re, os, shutil

# Download 바로 밑에 있는 csv로 끝나는 파일 혹은 폴더 목록
files = glob.glob("C:/Users/human/Downloads/*csv")
# Download 안 폴더 안에 csv로 끝나는 파일 혹은 폴더 목록
glob.glob("C:/Users/human/Downloads/**/*csv")
# Download/폴더/폴더 안에 csv로 끝나는 파일 혹은 폴더 목록
glob.glob("C:/Users/human/Downloads/**/**/*csv")
# Download 안에 존재하는 csv로 끝나는 파일 혹은 폴더 목록 모두
glob.glob(
    "C:/Users/human/Downloads/**/*csv",
    recursive=True)

# 수원이라는 단어를 포함하는 문자열 목록만 추출
# 수원|영등포 이런식으로 or 조건 가능
new_files = [i for i in files if 
             re.compile("수원").findall(i)]
os.path.basename(new_files[0]) # 파일명만 추출
os.path.dirname(new_files[0]) # 파일이 존재하는 디렉토리명만 
os.path.abspath(new_files[0]) # 절대경로로 변경 
# os.path.abspath("./수원7월데이터.csv")
os.getcwd() # 작업경로 문자열로 출력
# 작업 경로 변경
os.chdir("C:/Users/human/Downloads/새 폴더 (2)")

# new_files 목록을 현재 작업경로에 복제해오는 코드
[shutil.copy(i,os.path.basename(i))  for i in new_files]

