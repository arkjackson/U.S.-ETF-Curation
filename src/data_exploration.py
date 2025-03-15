import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 파일 'NH_CONTEST_DATA_ETF_HOLDINGS.csv' 로드
etf_data = pd.read_csv('NH_CONTEST_DATA_ETF_HOLDINGS.csv', encoding='latin1')

# ETF 티커 리스트
etf_tickers = ['VTI', 'SPY', 'QQQ']

# 그래프에 사용할 기간 설정
start_date = '2023-09-24'
end_date = '2024-09-24'

# 서브플롯 생성 (2행, 2열)
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

axes = axes.flatten()

# ETF에서 가장 높은 비중을 가진 상위 5개 주식을 찾기 위해 비중(가중치)으로 데이터 정렬
for i, etf_ticker in enumerate(etf_tickers):
    stock_data = etf_data[etf_data['etf_tck_cd'] == etf_ticker]
    top_etf_stocks = stock_data.sort_values(by='wht_pct', ascending=False).head(5)
    top_etf_stocks_tickers = top_etf_stocks[['tck_iem_cd', 'wht_pct']].values
    top_etf_stocks_tickers_list = [(t[0],t[1]) for t in top_etf_stocks_tickers]
    
    all_stock_data = pd.DataFrame()

    # 각 주식과 ETF 자체에 대한 주식 데이터를 다운로드
    for ticker, _ in top_etf_stocks_tickers_list + [(etf_ticker, None)]:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        all_stock_data[ticker] = stock_data['Close']
    
    # 데이터 정규화
    normalized_data = (all_stock_data - all_stock_data.min()) / (all_stock_data.max() - all_stock_data.min())
    
    # 해당 서브플롯(axes[i])에 그리기
    axes[i].plot(normalized_data.index, normalized_data[etf_ticker], label=etf_ticker, linewidth=3, color='black')
    
    # ETF의 상위 5개 주식 그리기
    for ticker, wht_pct in top_etf_stocks_tickers_list:
        label = f"{ticker} ({wht_pct:.2f}%)"
        axes[i].plot(normalized_data.index, normalized_data[ticker], label=label, linestyle='--')

    # 서브플롯 설정
    axes[i].set_title(f'Top 5 Stocks and {etf_ticker} ETF Normalized Prices (1 Year)')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Normalized Closing Price')
    axes[i].legend()
    axes[i].tick_params(axis='x', rotation=45)

# 레이아웃 조정 및 그래프 표시
plt.tight_layout()
plt.show()

# 서브플롯 생성 (2행, 2열)
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
axes = axes.flatten()

# 각 ETF에 대해 상위 5개 주식의 평균값 변화를 계산하고 그래프로 그리기
for i, etf_ticker in enumerate(etf_tickers):
    # 해당 ETF의 상위 5개 주식을 비중 순으로 정렬하여 선택
    stock_data = etf_data[etf_data['etf_tck_cd'] == etf_ticker]
    top_etf_stocks = stock_data.sort_values(by='wht_pct', ascending=False).head(5)
    top_etf_stocks_tickers = top_etf_stocks['tck_iem_cd'].tolist()
    
    all_stock_data = pd.DataFrame()

    # 상위 5개 주식과 ETF의 데이터를 다운로드
    for ticker in top_etf_stocks_tickers + [etf_ticker]:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        all_stock_data[ticker] = stock_data['Close']
    
    # 데이터 정규화
    normalized_data = (all_stock_data - all_stock_data.min()) / (all_stock_data.max() - all_stock_data.min())
    
    # 상위 5개 주식의 평균 계산
    top_stocks_mean = normalized_data[top_etf_stocks_tickers].mean(axis=1)
    
    # 해당 서브플롯에 ETF 가격과 상위 5개 주식 평균 가격 그리기
    axes[i].plot(normalized_data.index, normalized_data[etf_ticker], label=f'{etf_ticker} (ETF)', linewidth=3, color='black')
    axes[i].plot(normalized_data.index, top_stocks_mean, label='Top 5 Stocks Average', linewidth=2, color='blue', linestyle='--')
    
    # 서브플롯 설정
    axes[i].set_title(f'{etf_ticker} ETF vs. Top 5 Stocks Average (Normalized Prices)')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Normalized Closing Price')
    axes[i].legend()
    axes[i].tick_params(axis='x', rotation=45)

# 레이아웃 조정 및 그래프 표시
plt.tight_layout()
plt.show()