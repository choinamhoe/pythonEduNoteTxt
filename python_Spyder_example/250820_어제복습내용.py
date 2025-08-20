# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:02:56 2025

@author: human
"""

"""
re :  정규표현식(Regular Expression) 관련 패키지
shutil : 파일이나 폴더 전체를 복사 해 올 때 사용하는 패키지
os : 운영 체제 관련된 명령어들을 사용할 수 있는 패키지
    - 폴더 경로를 가져올 때 활용하거나
    - 작업 경로를 가져올 때 활용
glob: 정규표현식을 바탕으로 특정 파일 가져올 때 사용하는 패키지
"""
import re,shutil,os,glob
import datetime

now = datetime.datetime.now()
date = now.strftime("%Y%m%d")

print(dir(re))
print(dir(shutil))
print(dir(os))
print(dir(glob))

#파이썬개발에대한파일모음 폴더안에 csv파일 확장자 목록 가져옴
glob.glob("E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/*csv")
#파이썬개발에대한파일모음 폴더 아래에 폴더 안에 있는 csv파일 목록 가져옴
glob.glob("E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/**/*csv")
#파이썬개발에대한파일모음 폴더 아래에 존재하는 모든 csv파일 목록 가져옴
files = glob.glob(
            f"E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/{date}/*csv",
          recursive=True)
# 수원이라는 단어를 포함하는 문자열 목록만 추출
# 수원:영등포 이런식으로 or 조건 가능
new_files = [i for i in files if re.compile("수원").findall(i)]
new_files
os.path.basename(new_files[0]) #파일명만 추출
os.path.dirname(new_files[0]) #파일이 존재하는 디렉토리명만
os.path.abspath(new_files[0]) #절대경로로 변경
os.getcwd() #작업경로 문자열로 출력
#작업경로를 변경
os.chdir(f"E:/최남회/파이썬개발에대한파일모음/기상철 기상자료개발포털/{date}")

#new_files 목록을 현재 작업경로에 복제해오는 코드
[shutil.copy(i, os.path.basename(i)) for i in new_files ] 
