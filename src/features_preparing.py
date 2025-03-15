import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# 'NH_CONTEST_NHDATA_CUS_TP_IFO.csv' 데이터를 변환하는 함수
def transform_cus_tp_ifo():
    df = pd.read_csv('NH_CONTEST_NHDATA_CUS_TP_IFO.csv')

    # 'bse_dt'와 'tck_iem_cd'를 기준으로 데이터를 통합하기 위해 피벗
    pivot_df = df.pivot_table(
        index=['bse_dt', 'tck_iem_cd'],
        columns=['cus_cgr_llf_cd', 'cus_cgr_mlf_cd'],
        values=['cus_cgr_act_cnt_rt', 'cus_cgr_ivs_rt'],
        aggfunc='first'
    )

    # Flatten the column index for easier access
    pivot_df.columns = [f'{var}_{llf_cd}_{mlf_cd}' for var, llf_cd, mlf_cd in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # 결측값을 0으로 채움
    pivot_df = pivot_df.fillna(0)

    return pivot_df

# 'NH_CONTEST_NW_FC_STK_IEM_IFO.csv' 파일에 있는 주식 ticker, eng_name, 상장 주식 총량 데이터 가져오는 함수
def get_stockdata():
    df = pd.read_csv('NH_CONTEST_NW_FC_STK_IEM_IFO.csv', encoding='CP949')
    filtered_df = df[df['stk_etf_dit_cd'] == '주식'].drop_duplicates()
    return filtered_df['tck_iem_cd'].tolist(), filtered_df['fc_sec_eng_nm'].tolist(), filtered_df['ltg_tot_stk_qty'].tolist()

# Weighted Moving Average 계산 함수
def calculate_wma(data, window):
    weights = np.arange(1, window + 1)
    wma = data['Close'].rolling(window).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    return wma

# Momentum 계산 함수
def calculate_momentum(data, period):
    momentum = data['Close'] - data['Close'].shift(period)
    return momentum

# Stochastic %K 계산 함수
def calculate_stochastic_k(data, period=14):
    high_n = data['High'].rolling(window=period).max()
    low_n = data['Low'].rolling(window=period).min()
    stochastic_k = ((data['Close'] - low_n) / (high_n - low_n)) * 100
    return stochastic_k

# RSI 계산 함수 정의
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()  # 주가 변화 계산
    gain = delta.where(delta > 0, 0)  # 상승분 (양수만 저장)
    loss = -delta.where(delta < 0, 0)  # 하락분 (음수만 저장)

    avg_gain = gain.rolling(window=period).mean()  # 평균 상승분
    avg_loss = loss.rolling(window=period).mean()  # 평균 하락분

    rs = avg_gain / avg_loss  # 상대강도 계산 (RS)
    rsi = 100 - (100 / (1 + rs))  # RSI 공식

    return rsi

# LWR 계산 함수
def calculate_lwr(data, period=14):
    high_14 = data['High'].rolling(window=period).max()
    low_14 = data['Low'].rolling(window=period).min()

    # LWR 계산
    lwr = ((high_14 - data['Close']) / (high_14 - low_14)) * -100

    return lwr

# ADO 계산 함수
def calculate_ado(data):
    # Close Location Value
    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    ado = clv * data['Volume']

    return ado

# CCI 계산 함수
def calculate_cci(data, period=14):
    # Typical Price
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3

    # 14일 이동 평균 TP
    data['SMA_TP'] = data['TP'].rolling(window=period).mean()

    # 14일 이동 평균 편차
    data['Mean_Deviation'] = data['TP'].rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)

    # CCI 계산
    data['CCI'] = (data['TP'] - data['SMA_TP']) / (0.015 * data['Mean_Deviation'])

    return data

# MACD 계산 함수
def calculate_macd(data):
    # MACD 라인 계산
    data['MACD'] = data['EMA12'] - data['EMA26']

    # Signal Line 계산
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # MACD 히스토그램 계산
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    return data

# Turnover Ratio 계산 함수
def calculate_turnover(data, outstanding_shares):
    turnover = (data['Volume'] / outstanding_shares) * 100
    
    return turnover

# Weighted Close Price 계산 함수
def calculate_wcp(data):
    wcp = (data['High'] + data['Low'] + 2 * data['Close']) / 4

    return wcp

# Balance of Power 계산 함수
def calculate_bop(data):
    bop = (data['Close'] - data['Open']) / (data['High'] - data['Low'])

    return bop

file_paths = ['NH_CONTEST_STK_DT_QUT.csv','NH_CONTEST_NHDATA_STK_DD_IFO.csv']
file_names = ['STK_DT_QUT','STK_DD_IFO']

# 각 파일에서 추출할 열을 정의하는 딕셔너리
columns_to_extract = {
    'STK_DT_QUT': ['bse_dt','tck_iem_cd','iem_ong_pr','iem_hi_pr','iem_low_pr','iem_end_pr','acl_trd_qty'],
    'STK_DD_IFO': ['bse_dt', 'tck_iem_cd', 'tco_avg_pft_rt', 'lss_ivo_rt', 'pft_ivo_rt', 'tot_hld_act_cnt', 'tot_hld_qty', 'ifw_act_cnt', 'ofw_act_cnt', 'vw_tgt_cnt', 'rgs_tgt_cnt']
}

nh_data = pd.DataFrame()
# 파일을 순회하며 'bse_dt'와 'tck_iem_cd'를 기준으로 데이터를 병합
for idx, file_path in enumerate(file_paths):
    data = pd.read_csv(file_path)
    extracted_data = data[columns_to_extract[file_names[idx]]]
    if nh_data.empty:
        nh_data = extracted_data
    else:
        nh_data = pd.merge(nh_data, extracted_data, on=['bse_dt','tck_iem_cd'], how='inner')

# 변환된 'cus_tp_ifo' 데이터와 병합
transformed_cus_tp_ifo = transform_cus_tp_ifo()
nh_data = pd.merge(nh_data, transformed_cus_tp_ifo, on=['bse_dt','tck_iem_cd'], how='inner')

nh_data['tck_iem_cd'] = nh_data['tck_iem_cd'].str.strip()

ticker_symbols, stock_names, outstanding_shares = get_stockdata()

start_date = '2024-04-28'  
end_date = '2024-08-28'

data_frames = []

# 각 종목 코드를 순회
for index, ticker_symbol in enumerate(tqdm(ticker_symbols, desc="Processing tickers")):
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(interval='1d', start=start_date, end=end_date)

    # 주식 티커와 이름 추가
    df['TCK_IEM_CD'] = ticker_symbol
    df['ENG_NM'] = stock_names[index]

    # 단순 이동 평균(SMA) 계산
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()

    # 지수 이동 평균(EMA) 계산
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # 가중 이동 평균(WMA) 계산
    df['WMA5'] = calculate_wma(df, 5)
    df['WMA10'] = calculate_wma(df, 10)
    df['WMA20'] = calculate_wma(df, 20)

    # 모멘텀 계산
    df['Momentum5'] = calculate_momentum(df, 5)
    df['Momentum10'] = calculate_momentum(df, 10)
    df['Momentum20'] = calculate_momentum(df, 20)

    # 추가 지표 계산: %K, RSI, LWR, ADO, CCI, MACD, 회전율, 가중 종가, BOP
    df['%K'] = calculate_stochastic_k(df, period=14)
    df['RSI'] = calculate_rsi(df,period=14)
    df['LWR'] = calculate_lwr(df,period=14)
    df['ADO'] = calculate_ado(df) 
    df = calculate_cci(df, period=14)
    df = calculate_macd(df)
    df['Turnover'] = calculate_turnover(df, outstanding_shares[index])
    df['Weighted_Close'] = calculate_wcp(df)
    df['BOP'] = calculate_bop(df)
    
    # 원하는 날짜 범위 내의 데이터 필터링
    df = df.loc['2024-05-28':'2024-08-27']
    
    # 'Date' 열을 YYYYMMDD 형식으로 포맷팅
    df.index = pd.to_datetime(df.index).date
    df['Date'] = df.index 
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d').astype(int)

    # 출력할 관련 열 선택
    df = df[['Date','TCK_IEM_CD','ENG_NM','SMA5','SMA10','SMA20','EMA5','EMA10','EMA12','EMA20','EMA26','WMA5','WMA10','WMA20','Momentum5','Momentum10','Momentum20','%K','RSI','LWR','ADO','CCI','MACD_Histogram','Turnover','Weighted_Close','BOP']]

    data_frames.append(df)

# 모든 기술 지표 데이터를 하나의 데이터프레임으로 연결
technical_data = pd.concat(data_frames)

# 'nh_data'와 'technical_data'를 'bse_dt'와 'tck_iem_cd'를 기준으로 병합
merged_df = pd.merge(nh_data, technical_data, left_on=['bse_dt', 'tck_iem_cd'], right_on=['Date','TCK_IEM_CD'], how='inner')
merged_df = merged_df.sort_values(by=['tck_iem_cd','bse_dt'], ascending=True)
merged_df = merged_df.drop(columns=['Date', 'TCK_IEM_CD'])

# 'ENG_NM' 열을 'tck_iem_cd' 바로 다음에 위치하도록 열 순서 변경
cols = list(merged_df.columns)
cols.remove('ENG_NM')
eng_nm_index = cols.index('tck_iem_cd') + 1
cols.insert(eng_nm_index, 'ENG_NM')
merged_df = merged_df[cols]

# 총 1,123개의 주식 중 61일 기간 동안 데이터를 가진 1,110개의 주식을 필터링
date_range_by_ticker = merged_df.groupby('tck_iem_cd')['bse_dt'].nunique()
tickers_with_61_dates = date_range_by_ticker[date_range_by_ticker == 61].index
merged_df = merged_df[merged_df['tck_iem_cd'].isin(tickers_with_61_dates)]

# 결과를 CSV 파일로 저장
merged_df.to_csv('features.csv', index=False)