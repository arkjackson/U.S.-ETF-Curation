import pandas as pd
import pandas_market_calendars as mcal
from tqdm import tqdm
import yfinance as yf

def get_index_ticker(exchange_code):
    exchange_to_index_map = {
        "NYSE": "^GSPC",   # S&P 500 Index
        "NMS": "^IXIC",    # NASDAQ Composite Index
        "NYQ": "^DJI",     # Dow Jones Industrial Average
        "ASE": "^XAX",     # NYSE American Composite Index
        "NCM": "^IXIC",    # NASDAQ Composite Index (for NASDAQ Capital Market)
        "NGM": "^IXIC",    # NASDAQ Composite Index (for NASDAQ Global Market)
        "YHD": "^GSPC",    # S&P 500 (as a fallback for unknown exchanges)
        "BTS": "^IXIC",    # NASDAQ Composite (fallback)
        "PNK": "^DJI"      # Dow Jones Industrial Average (fallback for Pink Sheet stocks)
    }
    
    return exchange_to_index_map.get(exchange_code, None)

df = pd.read_csv('features.csv')

# 'bse_dt'를 datetime 형식으로 변환하고 'bse_dt'를 기준으로 중복 제거
df_copy = df.copy()
df_copy['bse_dt'] = pd.to_datetime(df_copy['bse_dt'], format='%Y%m%d')
df_unique_dates = df_copy.drop_duplicates(subset='bse_dt', keep='first').copy()

# NYSE 거래소의 유효한 거래일 가져오기 (최소 날짜와 최대 날짜 사이)
nyse = mcal.get_calendar('NYSE')
min_date = df_unique_dates['bse_dt'].min()
max_date = df_unique_dates['bse_dt'].max()
trading_days = nyse.valid_days(start_date=min_date, end_date=max_date + pd.Timedelta(days=1))
trading_days = trading_days.tz_localize(None)

# 현재 날짜 이후의 다음 거래일을 찾는 함수
def get_next_trading_day(current_date, trading_days):
    future_days = trading_days[trading_days > current_date]
    return future_days[0] if len(future_days) > 0 else pd.NaT
    
df_unique_dates['next_trading_day'] = df_unique_dates['bse_dt'].apply(lambda x: get_next_trading_day(x, trading_days))
next_trading_days_list = df_unique_dates['next_trading_day'].to_list()

# 각 종목에 대해 주식 및 시장 지수 데이터를 다운로드하고 처리
stock_tickers = df['tck_iem_cd'].unique()
data_frames = []

for ticker in tqdm(stock_tickers, desc="Processing tickers"):
    stock = yf.Ticker(ticker)
    info = stock.get_info()

    # 주식 및 시장 지수 데이터 다운로드
    market_index_ticker = get_index_ticker(info.get('exchange'))

    # 주식 및 시장 지수 데이터 다운로드
    start_date = '2024-05-28'
    end_date = '2024-08-28'
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    market_data = yf.download(market_index_ticker, start=start_date, end=end_date, progress=False)

    # 수익률 계산 및 다음 거래일에 따라 필터링
    stock_data['Return'] = stock_data['Close'].pct_change()
    market_data['Market Return'] = market_data['Close'].pct_change()
    stock_data = stock_data.loc[stock_data.index.isin(next_trading_days_list)]
    market_data = market_data.loc[market_data.index.isin(next_trading_days_list)]

     # 주식 및 시장 데이터를 병합하고, 연결을 위한 준비
    merged_data = pd.merge(stock_data[['Return']], market_data[['Market Return']], left_index=True, right_index=True)
    merged_data['tck_iem_cd'] = ticker
    data_frames.append(merged_data[['tck_iem_cd', 'Return', 'Market Return']])

# 모든 종목에 대한 데이터를 연결하고 초과 수익률 계산
target_df = pd.concat(data_frames)

# 거래일 리스트를 사용하여 하루 전 거래일을 찾는 함수
def get_previous_trading_day(current_date, trading_days):
    past_days = trading_days[trading_days < current_date]
    return past_days[-1] if len(past_days) > 0 else pd.NaT

# 'target_df'의 인덱스를 datetime으로 변환
target_df.index = pd.to_datetime(target_df.index, format='%Y%m%d')

# 하루 전 거래일로 인덱스를 변경
target_df['previous_trading_day'] = target_df.index.map(lambda x: get_previous_trading_day(x, trading_days))

# 인덱스를 'previous_trading_day'로 변경 후 포맷팅
target_df.index = target_df['previous_trading_day'].dt.strftime('%Y%m%d')

# 'previous_trading_day' 컬럼 삭제
target_df.drop(columns=['previous_trading_day'], inplace=True)

# 초과 수익률 계산
target_df['Excess Return'] = target_df['Return'] - target_df['Market Return']

# 초과 수익률을 기준으로 이진 타겟 생성
target_df['Target'] = target_df['Excess Return'].apply(lambda x: 1 if x > 0 else 0)

# 최종 데이터프레임을 CSV 파일로 저장
target_df.to_csv('target.csv')