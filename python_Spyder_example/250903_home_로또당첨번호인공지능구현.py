import os
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import numpy as np
import datetime
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================
# 1. 다가올 토요일 날짜
# ==========================
today = datetime.date.today()
days_until_saturday = (5 - today.weekday() + 7) % 7
upcoming_saturday = today + datetime.timedelta(days=days_until_saturday)
formatted_date = upcoming_saturday.strftime('%Y.%m.%d')

latest_round = 1205
save_folder = f"D:/lotto/{latest_round}회(당첨일-{formatted_date})"
os.makedirs(save_folder, exist_ok=True)

# ==========================
# 2. requests Session 설정
# ==========================
def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Connection": "close"
    })
    return session

# ==========================
# 3. JSON API 로또 수집
# ==========================
def crawl_lotto(max_round):
    print(f"📡 1회 ~ {max_round}회 로또 데이터 수집 시작")
    session = create_session()
    rows = []

    start_time = time.time()

    for i in range(1, max_round + 1):
        try:
            url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={i}"
            res = session.get(url, timeout=(5, 10))
            data = res.json()

            if data.get("returnValue") != "success":
                continue

            rows.append({
                "회차": i,
                "번호1": data["drwtNo1"],
                "번호2": data["drwtNo2"],
                "번호3": data["drwtNo3"],
                "번호4": data["drwtNo4"],
                "번호5": data["drwtNo5"],
                "번호6": data["drwtNo6"],
                "보너스": data["bnusNo"],
                "1등당첨복권수": data["firstPrzwnerCo"],
                "1등1개당첨금": data["firstWinamnt"]
            })

            if i % 50 == 0:
                elapsed = int(time.time() - start_time)
                print(f"⏳ {i}/{max_round}회 완료 ({elapsed}s)")

            time.sleep(0.1)

        except Exception as e:
            print(f"⚠️ {i}회차 수집 실패 → skip ({type(e).__name__})")
            time.sleep(0.5)

    print("✅ 데이터 수집 완료")
    return pd.DataFrame(rows)

# ==========================
# 4. 머신러닝 예측
# ==========================
def predict_lotto_numbers(df, n_predictions=5):
    X = df[['회차']].values
    y = df[[f'번호{i+1}' for i in range(6)]].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)

    next_round = df['회차'].max() + 1
    base_pred = np.round(model.predict([[next_round]])[0]).astype(int)

    predictions = set()
    while len(predictions) < n_predictions:
        noise = np.random.randint(-3, 4, size=6)
        nums = np.clip(base_pred + noise, 1, 45)
        nums = sorted(set(nums))
        while len(nums) < 6:
            nums.append(np.random.randint(1, 46))
            nums = sorted(set(nums))
        predictions.add(tuple(nums[:6]))

    return [list(p) for p in predictions]

# ==========================
# 5. 통계 기반 예측
# ==========================
def get_statistical_predictions(df, n_predictions=5):
    numbers = df[[f'번호{i+1}' for i in range(6)]].values.flatten()
    common = [n for n, _ in Counter(numbers).most_common(20)]

    preds = []
    for _ in range(n_predictions):
        preds.append(sorted(np.random.choice(common, 6, replace=False)))
    return preds

# ==========================
# 6. 메인 실행
# ==========================
if __name__ == "__main__":

    lotto_df = crawl_lotto(latest_round)

    if lotto_df.empty:
        print("❌ 데이터 수집 실패")
        exit()

    csv_path = f"{save_folder}/lotto_{latest_round}_회차.csv"
    lotto_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장 완료 → {csv_path}")

    print("🔮 전체 데이터 기반 예측")
    all_preds = predict_lotto_numbers(lotto_df, 3) + get_statistical_predictions(lotto_df, 2)

    recent_df = lotto_df.tail(50).reset_index(drop=True)
    print("🔮 최근 50회차 기반 예측")
    recent_preds = predict_lotto_numbers(recent_df, 3) + get_statistical_predictions(recent_df, 2)

    def save_txt(name, preds):
        path = os.path.join(save_folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{latest_round + 1}회차 로또 예측 번호\n\n")
            for i, nums in enumerate(preds, 1):
                f.write(f"추천 {i}: {nums}\n")
        print(f"✅ 저장 완료 → {path}")

    save_txt(f"[전체] 추천번호_{latest_round+1}회차.txt", all_preds)
    save_txt(f"[최근50] 추천번호_{latest_round+1}회차.txt", recent_preds)
