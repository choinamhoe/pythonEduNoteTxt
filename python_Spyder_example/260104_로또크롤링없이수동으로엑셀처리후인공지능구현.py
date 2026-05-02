import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import datetime

# =========================
# 0. 기본 설정
# =========================
LATEST_ROUND = 1222

today = datetime.date.today()
days_until_saturday = (5 - today.weekday() + 7) % 7
upcoming_saturday = today + datetime.timedelta(days=days_until_saturday)
formatted_date = upcoming_saturday.strftime('%Y.%m.%d')

BASE_DIR = rf"D:\github\pythonEduNoteTxt\lotto\{LATEST_ROUND}회(당첨일-{formatted_date})"
CSV_PATH = os.path.join(BASE_DIR, "lotto_history.csv")

os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# 1. 데이터 로드
# =========================
def load_lotto_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ CSV 파일 없음: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_cols = [f"번호{i}" for i in range(1, 7)]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("❌ CSV 컬럼 구조가 맞지 않습니다 (번호1~번호6 필요)")

    return df

# =========================
# 2. 머신러닝 기반 예측
# =========================
def predict_lotto_ml(df, n_predictions=3):
    X = df[['회차']].values
    y = df[[f'번호{i}' for i in range(1, 7)]].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    model.fit(X, y)

    next_round = df['회차'].max() + 1
    base_pred = np.round(model.predict([[next_round]])[0]).astype(int)

    predictions = []
    while len(predictions) < n_predictions:
        noise = np.random.randint(-3, 4, size=6)
        nums = np.clip(base_pred + noise, 1, 45)
        nums = sorted(set(nums))

        while len(nums) < 6:
            n = np.random.randint(1, 46)
            if n not in nums:
                nums.append(n)
        nums = sorted(nums[:6])

        if nums not in predictions:
            predictions.append(nums)

    return predictions

# =========================
# 3. 통계 기반 예측
# =========================
def predict_lotto_stat(df, n_predictions=2):
    nums = df[[f'번호{i}' for i in range(1, 7)]].values.flatten()
    common = [n for n, _ in Counter(nums).most_common(20)]

    predictions = []
    while len(predictions) < n_predictions:
        pick = sorted(np.random.choice(common, 6, replace=False).tolist())
        if pick not in predictions:
            predictions.append(pick)

    return predictions

# =========================
# 4. 결과 저장
# =========================
def save_result(filename, title, predictions):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=============== {LATEST_ROUND + 1}회차 로또 예측 번호 ===============\n\n")
        f.write(title + "\n")
        for i, nums in enumerate(predictions, 1):
            clean_nums = [int(n) for n in nums]
            f.write(f"추천 {i}: {clean_nums}\n")
        f.write("\n※ 본 결과는 통계/머신러닝 기반 참고용이며 당첨을 보장하지 않습니다.\n")

    print(f"✅ 저장 완료: {path}")

# =========================
# 5. 메인 함수
# =========================
def main():
    print("📂 로컬 CSV 기반 로또 분석 시작")

    df = load_lotto_data()

    # 전체 데이터
    all_ml = predict_lotto_ml(df, 3)
    all_stat = predict_lotto_stat(df, 2)
    all_preds = all_ml + all_stat

    save_result(
        f"[전체 데이터 기반] 추천 번호_{LATEST_ROUND + 1}_회차.txt",
        f"--- 전체 기간 (1~{LATEST_ROUND}회) 데이터 기반 ---",
        all_preds
    )

    # 최근 50회
    recent_df = df.tail(50).reset_index(drop=True)
    recent_ml = predict_lotto_ml(recent_df, 3)
    recent_stat = predict_lotto_stat(recent_df, 2)
    recent_preds = recent_ml + recent_stat

    save_result(
        f"[최근 50회 기반] 추천 번호_{LATEST_ROUND + 1}_회차.txt",
        "--- 최근 50회차 데이터 기반 ---",
        recent_preds
    )

    print("🎉 모든 작업 완료")

# =========================
# 실행
# =========================
if __name__ == "__main__":
    main()
