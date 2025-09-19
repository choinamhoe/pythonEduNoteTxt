# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 17:37:00 2025

@author: human
"""

import os
import requests
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime
%matplotlib auto

# ==========================
# 1.다가올 토요일 일자 구하기
# ==========================
# 1. 오늘 날짜 가져오기
today = datetime.date.today()
# 예시 날짜로 테스트하려면 아래 주석을 해제하세요.
# today = datetime.date(2025, 9, 8)

# 2. 오늘 요일 계산 (월요일=0, 화요일=1, ..., 토요일=5, 일요일=6)
#    다가오는 토요일(5)까지 남은 날짜 계산
days_until_saturday = (5 - today.weekday() + 7) % 7

# 3. 오늘 날짜에 남은 날짜를 더해 다음 토요일 날짜 계산
upcoming_saturday = today + datetime.timedelta(days=days_until_saturday)

# 4. 'YYYY.MM.DD' 형식으로 날짜 포맷 변경
formatted_date = upcoming_saturday.strftime('%Y.%m.%d')

# 결과 출력
#print(formatted_date)
# ==========================
# 2. 로또 당첨번호 크롤링
# ==========================

number = 1188
base_folder = f"E:/choinamhoe/lotto/{number}회(당첨일-{formatted_date})"
# 폴더가 없다면 자동 생성
os.makedirs(base_folder, exist_ok=True)

def get_lotto_numbers(drwNo):
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}"
    res = requests.get(url).json()
    if res["returnValue"] != "success":
        return None
    nums = [res[f"drwtNo{i}"] for i in range(1, 7)]
    bonus = res["bnusNo"]
    return {
        "회차": drwNo,
        "번호1": nums[0],
        "번호2": nums[1],
        "번호3": nums[2],
        "번호4": nums[3],
        "번호5": nums[4],
        "번호6": nums[5],
        "보너스": bonus,
    }

def crawl_lotto(max_round=number):
    data = []
    for i in range(1, max_round+1):
        result = get_lotto_numbers(i)
        if result:
            data.append(result)
    df = pd.DataFrame(data)
    df.to_csv(f"{base_folder}/lotto_{number}_회차.csv", index=False, encoding="utf-8-sig")
    return df

# ==========================
# 3. 데이터 분석
# ==========================
def analyze_lotto(df, recent_n=None):
    """전체 or 최근 n회 로또 데이터 분석"""
    if recent_n:
        df = df.tail(recent_n)  # 최근 N회만 사용
    numbers = []
    for i in range(1, 7):
        numbers.extend(df[f"번호{i}"].tolist())
    freq = pd.Series(numbers).value_counts().sort_index()
    return freq

# ==========================
# 4. 확장된 추천번호 생성
# ==========================
def recommend_numbers(freq, n=6):
    prob = freq / freq.sum()

    while True:
        nums = sorted(random.choices(list(prob.index), weights=prob.values, k=n))
        
        # (조건1) 홀짝 비율 체크 (2:4 ~ 4:2 사이)
        odd = sum(1 for x in nums if x % 2 == 1)
        even = n - odd
        if not (2 <= odd <= 4):
            continue
        
        # (조건2) 구간 분포 체크 (1~10, 11~20, 21~30, 31~40, 41~45)
        sections = [0, 0, 0, 0, 0]
        for x in nums:
            if 1 <= x <= 10: sections[0] += 1
            elif 11 <= x <= 20: sections[1] += 1
            elif 21 <= x <= 30: sections[2] += 1
            elif 31 <= x <= 40: sections[3] += 1
            else: sections[4] += 1
        if sum(1 for s in sections if s > 0) < 3:  # 최소 3구간 이상 포함
            continue

        return nums

# ==========================
# 5. 실행
# ==========================
if __name__ == "__main__":
    print("📌 로또 데이터 수집중...")
    df = crawl_lotto(number)   # 최신 회차까지 크롤링
    print("✅ 데이터 저장 완료: lotto.csv")

    # 전체 데이터 기반 분석
    freq_all = analyze_lotto(df)
    print("\n📊 전체 데이터 기반 번호 출현 빈도")
    print(freq_all)
    
    recent_n=50
    # 최근 50회 데이터 기반 분석
    freq_recent = analyze_lotto(df, recent_n)
    print("\n📊 최근 50회 데이터 기반 번호 출현 빈도")
    print(freq_recent)
    
    # 추천 번호 (전체 데이터 기반)
    file_name_all = os.path.join(base_folder, f"[전체 데이터 기반] 추천 번호_{number + 1}.txt")
   
    print("\n🎯 [전체 데이터 기반] 추천 번호")
    for i in range(5):
        print(f"추천 {i+1}:", recommend_numbers(freq_all))
    with open(file_name_all, "w", encoding="utf-8") as f:   # UTF-8로 저장
        for i in range(5):
            numbers = recommend_numbers(freq_all)   # 한 번만 호출
            line = f"추천 {i+1}: {numbers}"
            print(line)         # 콘솔 출력
            f.write(line + "\n")  # 파일에 저장

    # 추천 번호 (최근 50회 기반)
    print("\n🎯 [최근 50회 기반] 추천 번호")
    # 파일 전체 경로
    file_name_recent = os.path.join(base_folder, f"[최근 50회 기반] 추천 번호_{number + 1}.txt")
    
    with open(file_name_recent, "w", encoding="utf-8") as f:   # UTF-8로 저장
        for i in range(5):
            numbers = recommend_numbers(freq_recent)   # 한 번만 호출
            line = f"추천 {i+1}: {numbers}"
            print(line)         # 콘솔 출력
            f.write(line + "\n")  # 파일에 저장

    # 그래프 출력
    #freq_all.plot(kind="bar", figsize=(12,5), title="로또 번호 출현 빈도 (전체)")
    #plt.show()

    #freq_recent.plot(kind="bar", figsize=(12,5), color="orange", title="로또 번호 출현 빈도 (최근 50회)")
    #plt.show()