import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

filtered_xgb_common_features = xgb_common_features[xgb_common_features['feature_name'].isin(candidate_features)]

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

n_markovian = 5

# 최근 n번째 이전 날짜를 찾기
start_date = unique_dates[-1 * n_markovian]

# 특정 기간의 데이터 필터링
tmp_features_target = data_features_target[data_features_target['bse_dt'] >= start_date]

etf_data = pd.read_csv('NH_CONTEST_DATA_ETF_HOLDINGS.csv', encoding='cp949')

# Filter rows where 'sec_tp' is 'ST' (which represents stocks)
stock_data = etf_data[etf_data['sec_tp'] == 'ST']

# Group the stocks by 'etf_tck_cd' (ETF ticker code) and collect both stock tickers and their weights
stocks_by_etf = stock_data.groupby('etf_tck_cd').apply(lambda x: list(zip(x['tck_iem_cd'], x['wht_pct']))).to_dict()

etf_list = list(stocks_by_etf.keys())
etf_price_data = yf.download(etf_list, start='2024-08-27', end='2024-09-25', progress=False)['Close']
etf_price_data.index = pd.to_datetime(etf_price_data.index).date # Remove the time component from the price_data index

predictions_dict = {}

# portfolio manager
portfolio = {}

# Initialize a list to store daily portfolio values
daily_values = []

cash = 1000000 # 1,000,000 dollars 

for date, group in testing_data_grouped:
    
    # Initialize a dictionary to store the weighted average predictions per ETF
    etf_weighted_avg_predictions = {}

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42
    )

    X_train = tmp_features_target[filtered_xgb_common_features['feature_name']]
    y_train = tmp_features_target['Target']
    
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]

    rf.fit(X_train, y_train)

    X_test = group[filtered_xgb_common_features['feature_name']]
    y_test = group['Target']

    X_test = X_test.dropna()  
    y_test = y_test[X_test.index]

    # TCK_IEM_CD도 X의 인덱스에 맞게 선택
    tck_iem_cd = group.loc[X_test.index, 'tck_iem_cd']

    date = date.date()

    predictions = rf.predict(X_test)

    predictions_dict[date] = pd.DataFrame({
        'tck_iem_cd': tck_iem_cd,  # Stock ticker
        'Prediction': predictions           # Model's predicted values
    })

    for etf, stocks_weights in stocks_by_etf.items():
        weighted_sum = 0
        for stock, weight in stocks_weights:
            if stock in predictions_dict[date]['tck_iem_cd'].values:
                stock_prediction = predictions_dict[date][predictions_dict[date]['tck_iem_cd'] == stock]['Prediction'].values[0]
            else:
                stock_prediction = 0
            weighted_sum += stock_prediction * weight
            
        etf_weighted_avg_predictions[etf] = weighted_sum

    filtered_predictions_buy = {etf: pred for etf, pred in etf_weighted_avg_predictions.items() if pred > 50}
    filtered_predictions_buy= dict(sorted(filtered_predictions_buy.items(), key=lambda item: item[1], reverse=True)) # 확률 큰 것부터 금액 비중 크게 매수
    
    filtered_predictions_sell = {etf: pred for etf, pred in etf_weighted_avg_predictions.items() if pred < 45}

    current_etf_prices = etf_price_data.loc[date]

    total_numbers = len(filtered_predictions_buy) + len(filtered_predictions_sell)

    # 투자 금액 조정 -> 상승장, 하락장에 따라 투자 금액 조절
    if len(filtered_predictions_buy) >= int(total_numbers * 0.5):
        buy_cost = cash
    else:
        buy_cost = cash * 0.5

    total_pred_sum = sum(filtered_predictions_buy.values())

    # Buy Signal
    for etf, pred in filtered_predictions_buy.items():
        if etf not in portfolio:
            price = current_etf_prices.get(etf, None)
            if price:
                invest_ratio = pred / total_pred_sum

                # 예상 확률에 따라 가중치 다르게 부여
                if pred >= 90:
                    invest_ratio *= 3
                elif pred >= 80:
                    invest_ratio *= 2.5
                elif pred >= 70:
                    invest_ratio *= 2
                elif pred >= 60:
                    invest_ratio *= 1.5

                invest_amount = buy_cost * invest_ratio
                quantity = invest_amount / price
                cash -= invest_amount
                portfolio[etf] = {'quantity': quantity, 'buy_price': price}
    
    # Sell Signal
    for etf, pred in filtered_predictions_sell.items():
        if etf in portfolio:
            price = current_etf_prices.get(etf, None)
            if price:
                quantity = portfolio[etf]['quantity']
                cash += quantity * price
                del portfolio[etf]

    # Update portfolio value based on current prices
    portfolio_value = sum(current_etf_prices.get(etf, 0) * info['quantity'] for etf, info in portfolio.items())
    total_value = cash + portfolio_value

    # Log the portfolio value and cash for the day
    print(f"{date}: Cash: {cash:.2f}, Portfolio Value: {portfolio_value:.2f}, Total Value: {total_value:.2f}")

    # Log the daily total value
    daily_values.append({'Date': date, 'Total Value': total_value})

    # 종목마다 가장 오래된 데이터 삭제
    tmp_features_target = tmp_features_target.sort_values('bse_dt').groupby('tck_iem_cd').apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    
    # 기존 데이터와 새로운 데이터를 합쳐서 정렬
    tmp_features_target = pd.concat([tmp_features_target, group]).sort_values(['tck_iem_cd','bse_dt']).reset_index(drop=True)
