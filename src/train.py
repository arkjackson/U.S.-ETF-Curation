import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

filtered_xgb_common_features = xgb_common_features[xgb_common_features['feature_name'].isin(candidate_features)]

# testing models & evaluate models
def update_and_predict(models, X_new, y_new, metrics_storage):

    for model_name, model in models.items():
        y_pred = model.predict(X_new)
        
        # 평가 지표 계산
        f1 = f1_score(y_new, y_pred, average='binary')

        metrics_storage[model_name]['f1'].append(f1)

# GA-XGBoost -> Model Train
data_features_target = pd.read_csv('features_target.csv')

# 'bse_dt' 컬럼을 datetime 형식으로 변환
data_features_target['bse_dt'] = pd.to_datetime(data_features_target['bse_dt'], format='%Y%m%d')

# 고유한 날짜를 정렬
unique_dates = data_features_target['bse_dt'].sort_values().unique()

testing_data = pd.read_csv('testing_data.csv')

# 컬럼 이름 변경 ('Date' -> 'bse_dt', 'TCK_IEM_CD' -> 'tck_iem_cd')
testing_data = testing_data.rename(columns={'Date': 'bse_dt', 'TCK_IEM_CD': 'tck_iem_cd'})

testing_data['bse_dt'] = pd.to_datetime(testing_data['bse_dt'], format='%Y%m%d')

# 겹치는 종목 코드 찾기
common_tickers = set(data_features_target['tck_iem_cd']).intersection(set(testing_data['tck_iem_cd']))

# 겹치는 종목만 남기고 제거
testing_data = testing_data[testing_data['tck_iem_cd'].isin(common_tickers)]

testing_data_grouped  = testing_data.groupby('bse_dt')

all_metrics_storage = {}
all_date_lists = {}

markovian_list = [20,15,10,5]
for n_markovian in tqdm(markovian_list):
    metrics_storage = {
        "XGBoost": {'f1': []},
        "RandomForest": {'f1': []},
        "LightGBM": {'f1': []},
        "SupportVectorMachine": {'f1': []}
    }

    models = {
        "XGBoost": xgb,
        "RandomForest": rf,
        "LightGBM": lgbm,
        "SupportVectorMachine": svm
    }

    # 최근 n번째 이전 날짜를 찾기
    start_date = unique_dates[-1 * n_markovian]

    # 특정 기간의 데이터 필터링
    tmp_features_target = data_features_target[data_features_target['bse_dt'] >= start_date]

    date_list = []

    for date, group in tqdm(testing_data_grouped, desc='Processing train'):
        # Model Setting
        xgb_params = {
            'objective': 'binary:logistic',  # 이진 분류 문제이므로 'binary:logistic' 사용
            'learning_rate': 0.05,
            'max_depth': 3,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'gamma': 1,
            'lambda': 1,
            'seed': 42,
            'eval_metric': ['logloss','auc']
        }

        xgb = XGBClassifier(**xgb_params)

        # RandomForest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=5,
            random_state=42
        )

        # LightGBM
        lgbm = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=300,
            num_leaves=7,
            max_depth=3,
            min_child_samples=10,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            lambda_l1=0.1,
            lambda_l2=0.1,
            random_state=42
        )

        # SVM
        svm = SVC(
            kernel='rbf',
            C=1,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        )
        
        date_list.append(date)

        X_train = tmp_features_target[filtered_xgb_common_features['feature_name']]
        y_train = tmp_features_target['Target']
        
        X_train = X_train.dropna()
        y_train = y_train[X_train.index]

        xgb.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)
        svm.fit(X_train, y_train)        

        X_test = group[filtered_xgb_common_features['feature_name']]
        y_test = group['Target']

        X_test = X_test.dropna()  
        y_test = y_test[X_test.index]

        update_and_predict(models, X_test, y_test, metrics_storage)

        # 종목마다 가장 오래된 데이터 삭제
        tmp_features_target = tmp_features_target.sort_values('bse_dt').groupby('tck_iem_cd').apply(lambda x: x.iloc[1:]).reset_index(drop=True)
        
        # 기존 데이터와 새로운 데이터를 합쳐서 정렬
        tmp_features_target = pd.concat([tmp_features_target, group]).sort_values(['tck_iem_cd','bse_dt']).reset_index(drop=True)

    # 각 n-markovian에 대한 결과 저장
    all_metrics_storage[n_markovian] = metrics_storage
    all_date_lists[n_markovian] = date_list

    print(f"{n_markovian}-Markovian Property Applied")
    for model, metrics in metrics_storage.items():
        for metric_name, values in metrics.items():
            mean = np.mean(values)
        print(model, mean)