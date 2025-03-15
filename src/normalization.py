import pandas as pd

# Min-Max 정규화를 적용하는 함수
def min_max_normalize(group, columns):
    for column in columns:
        min_val = group[column].min()
        max_val = group[column].max()
        group[column] = (group[column] - min_val) / (max_val - min_val) if max_val > min_val else 0
    return group

df_x = pd.read_csv('features.csv')
feature_columns = df_x.columns.difference(['bse_dt', 'tck_iem_cd', 'ENG_NM', 'Target'])

# 'tck_iem_cd'를 기준으로 그룹화하여 Min-Max 정규화 적용
normalized_df = df_x.groupby('tck_iem_cd').apply(lambda group: min_max_normalize(group, feature_columns))

normalized_df = normalized_df.reset_index(drop=True)

df_y = pd.read_csv('target.csv')

# feature 데이터와 target데이터 병합 및 불필요한 컬럼 삭제
merged_df = pd.merge(normalized_df, df_y, left_on=['bse_dt', 'tck_iem_cd'], right_on=['previous_trading_day', 'tck_iem_cd'], how='inner')
merged_df = merged_df.drop(columns=['previous_trading_day', 'Return', 'Market Return', 'Excess Return'])

final_columns = list(normalized_df.columns) + ['Target']

merged_df = merged_df[final_columns]

# 최종 데이터프레임을 'features_target.csv' 파일로 저장
merged_df.to_csv('features_target.csv', index=False)