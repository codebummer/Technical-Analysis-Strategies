import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
from datetime import datetime 
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

start = datetime(2010, 7, 23)
end = datetime.today()
years = end.year - start.year + 0.5

yf.pdr_override()
def fetch_prices(stocks):
    df = pd.DataFrame([])
    for ticker, stock in stocks.items():
         df[stock] = (web.get_data_yahoo(ticker, start, end)['Close'])
    return df

stocks = {
  'SPY' : 'US Stocks',
  'EFA' : 'Non-US Dveloped Market Stocks',
  'EEM' : 'Emerging Market Stocks',
  'DBC' : 'Commodities',
  'GLD' : 'Gold',
  'EDV' : 'Extended Duration Teasuries',
  'LTPZ' : 'Tresuary Inflation-Protected Securities',
  'LQD' : 'US Corporate Bonds',
  'EMLC' : 'Emerging Market Bonds'
}

prices = fetch_prices(stocks)

with sqlite3.connect('allweather_portfolio.db') as db:
  prices.to_sql('Stock_Prices', db, if_exists='replace')

BIZDAYS_A_YEAR = 252
daily_ret = prices.pct_change().add(1).cumprod()
annual_ret = np.power(daily_ret[-1:], 1/years) - 1
daily_cov = prices.pct_change().cov()
annual_cov = daily_cov * BIZDAYS_A_YEAR
stds = prices.rolling(window=BIZDAYS_A_YEAR, min_periods=1).std()
peaks = prices.rolling(window=BIZDAYS_A_YEAR, min_periods=1).max()
drawdowns = prices/peaks - 1
mdds = drawdowns.rolling(window=BIZDAYS_A_YEAR, min_periods=1).min().min()

cagr = annual_ret.iloc[0].values.sum()
max_risk = stds.max().mean()
sharpe = (cagr-annual_ret['Tresuary Inflation-Protected Securities'].values[-1])/max_risk
mdd = mdds.mean()

print(f'CAGR: {cagr}, Max Risk: {max_risk}, Sharpe Ratio: {sharpe}, MDD: {mdd}')
