import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import numpy as np
from tqdm import tqdm
import datetime
import time  # time 모듈 추가

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

latest_round = 1190
save_folder = f"E:/choinamhoe/lotto/{latest_round}회(당첨일-{formatted_date})"
# 폴더가 없다면 자동 생성
os.makedirs(save_folder, exist_ok=True)

def get_lotto_numbers(drwNo):
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}"
    res = requests.get(url , timeout=10).json()
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

def crawl_lotto(max_round=latest_round):
    data = []
    for i in range(1, max_round+1):
        result = get_lotto_numbers(i)
        if result:
            data.append(result)
    
    # ✅ 2. 각 요청 사이에 0.5초의 딜레이를 줌
    time.sleep(0.5)
        
    df = pd.DataFrame(data)
    #df.to_csv(f"{save_folder}/lotto_{latest_round}_회차.csv", index=False, encoding="utf-8-sig")
    return df

def get_lotto_win_numbers(start_round, end_round):
    """
    동행복권 사이트에서 지정된 회차의 로또 당첨번호를 가져옵니다.
    """
    results = []
    print(f"{start_round}회부터 {end_round}회까지 당첨번호를 가져옵니다...")
    for round_num in tqdm(range(start_round, end_round + 1)):
        try:
            url = f"https://www.dhlottery.co.kr/gameResult.do?method=byWin&drwNo={round_num}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # 요청 실패 시 예외 발생
            soup = BeautifulSoup(response.text, "html.parser")

            win_numbers_div = soup.find("div", class_="win_result")
            if not win_numbers_div:
                continue
                
            win_numbers = [int(p.text) for p in win_numbers_div.select("div.ball_645.lrg span")]
            bonus_num = int(win_numbers_div.select_one("div.bonus p.ball_645.lrg span").text)
            
            draw_date = soup.select_one("p.desc").text.replace("추첨", "").strip()

            row = {'회차': round_num, '추첨일': draw_date}
            for i in range(6):
                row[f'번호{i+1}'] = win_numbers[i]
            row['보너스'] = bonus_num
            results.append(row)
            
            # ✅ 2. 각 요청 사이에 0.5초의 딜레이를 줌
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"{round_num}회차 정보를 가져오는 중 오류 발생: {e}")
        except Exception as e:
            print(f"{round_num}회차 데이터 파싱 중 오류 발생: {e}")

    return pd.DataFrame(results)

def predict_lotto_numbers(df, n_predictions=5):
    """
    Random Forest 모델을 사용하여 다음 로또 번호를 예측합니다.
    """
    if len(df) < 2:
        return [list(range(1, 7))] * n_predictions # 데이터가 부족할 경우 기본 번호 반환

    X = df[['회차']].values
    y = df[[f'번호{i+1}' for i in range(6)]].values

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    next_round = df['회차'].max() + 1
    predicted_numbers = model.predict([[next_round]])
    
    # 예측된 번호들을 가장 가까운 정수로 변환하고 중복 제거
    predicted_sets = []
    for _ in range(n_predictions * 5): # 다양한 예측 생성을 위해 더 많이 시도
        if len(predicted_sets) >= n_predictions:
            break
            
        base_prediction = np.round(predicted_numbers[0]).astype(int)
        
        # 약간의 무작위성을 추가하여 다양한 조합 생성
        noise = np.random.randint(-2, 3, size=6)
        new_prediction = base_prediction + noise
        
        # 번호가 1~45 범위 내에 있도록 조정 및 중복 제거
        new_prediction = np.clip(new_prediction, 1, 45)
        unique_numbers = sorted(list(set(new_prediction)))
        
        # 6개 번호가 되도록 조정
        while len(unique_numbers) < 6:
            extra_num = np.random.randint(1, 46)
            if extra_num not in unique_numbers:
                unique_numbers.append(extra_num)
            unique_numbers = sorted(list(set(unique_numbers)))
        
        final_set = sorted(unique_numbers[:6])
        if final_set not in predicted_sets:
            predicted_sets.append(final_set)
            
    return predicted_sets[:n_predictions]


def get_statistical_predictions(df, n_predictions=5):
    """
    통계적 빈도 분석을 기반으로 번호를 예측합니다.
    """
    numbers = df[[f'번호{i+1}' for i in range(6)]].values.flatten()
    most_common = [num for num, count in Counter(numbers).most_common(15)]
    
    predicted_sets = []
    for _ in range(n_predictions):
        # 가장 흔한 번호 중에서 무작위로 6개를 선택
        prediction = sorted(np.random.choice(most_common, 6, replace=False).tolist())
        predicted_sets.append(prediction)
        
    return predicted_sets


if __name__ == "__main__":
   
    print("📌 로또 데이터 수집중...")
    # 2. 1회부터 입력된 회차까지 당첨번호 스크래핑
    #lotto_df = get_lotto_win_numbers(1, latest_round)
    lotto_df = crawl_lotto(latest_round)   # 최신 회차까지 크롤링
    print(f"✅ 데이터 수집 완료 ... {lotto_df}")

    if not lotto_df.empty:
        # 3. 엑셀(CSV) 파일로 저장
        csv_filename = f'{save_folder}/lotto_{latest_round}_회차.csv'
        lotto_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n✅ 당첨번호가 '{csv_filename}' 파일로 저장되었습니다.")

        # 4. 전체 기간 데이터로 예측
        print("\n🔮 전체 기간 데이터 기반 예측 번호 생성 중...")
        # 머신러닝 예측과 통계적 예측을 혼합
        all_time_preds_ml = predict_lotto_numbers(lotto_df, n_predictions=3)
        all_time_preds_stat = get_statistical_predictions(lotto_df, n_predictions=2)
        all_time_predictions = all_time_preds_ml + all_time_preds_stat

        # 5. 최근 50회차 데이터로 예측
        print("🔮 최근 50회차 데이터 기반 예측 번호 생성 중...")
        recent_50_df = lotto_df.tail(50).reset_index(drop=True)
        recent_preds_ml = predict_lotto_numbers(recent_50_df, n_predictions=3)
        recent_preds_stat = get_statistical_predictions(recent_50_df, n_predictions=2)
        recent_predictions = recent_preds_ml + recent_preds_stat

        # 6. 예측 결과를 텍스트 파일로 저장
        #전체에 대한 파일 저장
        txt_filename_all_time = os.path.join(save_folder, f"[전체 데이터 기반] 추천 번호_{latest_round + 1}_회차.txt")
       
        with open(txt_filename_all_time, 'w', encoding='utf-8') as f:
            f.write(f"=============== {latest_round + 1}회차 로또 예측 번호 ===============\n\n")
            f.write(f"--- 전체 기간(1회 ~ {latest_round}회) 데이터 기반 예측 ---\n")
            for i, nums in enumerate(all_time_predictions):
                f.write(f"  추천 {i+1}: {str([int(n) for n in sorted(nums)])}\n")
            
            f.write("\n\n*주의: 본 예측은 통계적 및 머신러닝 분석에 기반한 것으로, 당첨을 보장하지 않습니다.\n")
        
        print(f"✅ 전체 기간(1회 ~ {latest_round}회) 예측 번호가 '{txt_filename_all_time}' 파일로 저장되었습니다.")
        
        #50회차에 대한 파일 저장
        txt_filename_recent = os.path.join(save_folder, f"[최근 50회 기반] 추천 번호_{latest_round + 1}_회차.txt")
       
        
        with open(txt_filename_recent, 'w', encoding='utf-8') as f:
             f.write(f"=============== {latest_round + 1}회차 로또 예측 번호 ===============\n\n")
            
             f.write("\n--- 최근 50회차 데이터 기반 예측 ---\n")
             for i, nums in enumerate(recent_predictions):
                 f.write(f"  추천 {i+1}: {str([int(n) for n in sorted(nums)])}\n")
             
             f.write("\n\n*주의: 본 예측은 통계적 및 머신러닝 분석에 기반한 것으로, 당첨을 보장하지 않습니다.\n")
         
        print(f"✅ 최근 50회차 예측 번호가 '{txt_filename_recent}' 파일로 저장되었습니다.")

    else:
        print("\n데이터를 가져오지 못했습니다. 인터넷 연결 및 입력한 회차를 확인해주세요.")