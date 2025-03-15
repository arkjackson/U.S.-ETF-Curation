import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# GA-ML(XGBoost) 실행 
random.seed(42)
np.random.seed(42)
feature_columns = merged_df.columns[~merged_df.columns.isin(['bse_dt', 'tck_iem_cd', 'ENG_NM', 'Target'])]
target_column = 'Target'

# GA-XGBoost 초기 설정
xgb_count_feature_idx_num = [0 for _ in range(len(feature_columns))] # 선택된 feature 횟수를 기록할 리스트

total_number_stock = len(merged_df.groupby('tck_iem_cd')) # 종목의 총 개수
xgb_min_loss = 1000 # 최소 손실 초기값 설정
xgb_total_loss = 0

# 각 종목(tck_iem_cd)별로 특성 선택을 수행
for ticker, group in tqdm(merged_df.groupby('tck_iem_cd'), desc="Processing Feature Selection"):
    X = np.array(group[feature_columns])
    y = np.array(group[target_column])
    
    # XGBClassifier 모델 초기화
    model_XGB = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'  # logloss를 사용하여 평가
    )

    # GA-ML 알고리즘을 사용하여 최적의 손실 및 선택된 특성 인덱스 찾기
    best_loss_xgb, best_feature_idx = show_result(X, y, model_XGB, ticker)
    xgb_total_loss += best_loss_xgb
    if best_loss_xgb < xgb_min_loss:
        xgb_min_loss = best_loss_xgb

    # 선택된 feature의 인덱스를 기록
    for idx, v in enumerate(best_feature_idx):
        if v == 1:
            xgb_count_feature_idx_num[idx] += 1

# 최종 feature 중요도 순위 계산
xgb_importance_feature_ranking = calculate_final_importance(total_number_stock)

# GA-XGBoost 결과 출력
print("Feature Engineering Completed!")
print("GA-XGBoost Algorithm Results")
xgb_results = {
    'Best Log Loss': round(xgb_min_loss, 5), # 최적의 Log Loss
    'Mean Best Log Loss': round(xgb_total_loss / total_number_stock, 5), # 평균 최적 Log Loss
    'Total Selected Features': xgb_count_feature_idx_num, # 선택된 feature 수
    'Feature Importance Ranking': xgb_importance_feature_ranking # feature 중요도 순위
}

for key, value in xgb_results.items():
    if isinstance(value, pd.DataFrame):  # DataFrame일 경우 별도로 출력
        print(f"{key}:")
        print(value)
    else:
        print(f"{key}: {value}")

# 결과 저장
with open('ga_xgb_results.txt', 'w') as f:
    for key, value in xgb_results.items():
        f.write(f'{key},{value}\n')

# GA-ML(LightGBM) 실행

feature_importances_global = None
lgb_count_feature_idx_num = [0 for _ in range(len(feature_columns))] # 선택된 feature 횟수를 기록할 리스트

total_number_stock = len(merged_df.groupby('tck_iem_cd'))
lgb_min_loss = 1000 # 최소 손실 초기값 설정
lgb_total_loss = 0

# 각 종목(tck_iem_cd)별로 feature selection 수행
for ticker, group in tqdm(merged_df.groupby('tck_iem_cd'), desc="Processing Feature Selection"):
    X = np.array(group[feature_columns])
    y = np.array(group[target_column])
    
    # LightGBM 모델 초기화
    model_LGB = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=20,
        max_depth=4,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    # GA-ML 알고리즘을 사용하여 최적의 손실 및 선택된 feature 인덱스 찾기
    best_loss_lgb, best_feature_idx = show_result(X, y, model_LGB, ticker)
    lgb_total_loss += best_loss_lgb
    if best_loss_lgb < lgb_min_loss:
        lgb_min_loss = best_loss_lgb

    # 선택된 feature의 인덱스를 기록
    for idx, v in enumerate(best_feature_idx):
        if v == 1:
            lgb_count_feature_idx_num[idx] += 1

# 최종 feature 중요도 순위 계산
lgb_importance_feature_ranking = calculate_final_importance(total_number_stock)

# GA-LightGBM 결과 출력

print("Feature Engineering Completed!")
print("GA-LightGBM Algorithm Results")

lgb_results = {
    'Best Log Loss': round(lgb_min_loss, 5), # 최적의 Log Loss
    'Mean Best Log Loss': round(lgb_total_loss / total_number_stock, 5), # 평균 최적 Log Loss
    'Total Selected Features': lgb_count_feature_idx_num, # 선택된 특성 수
    'Feature Importance Ranking': lgb_importance_feature_ranking # 특성 중요도 순위
}

for key, value in lgb_results.items():
    if isinstance(value, pd.DataFrame):  # DataFrame일 경우 별도로 출력
        print(f"{key}:")
        print(value)
    else:
        print(f"{key}: {value}")

# 결과 저장
with open('ga_lgb_results.txt', 'w') as f:
    for key, value in lgb_results.items():
        f.write(f'{key},{value}\n')