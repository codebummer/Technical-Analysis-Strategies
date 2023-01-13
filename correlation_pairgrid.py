import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime
import sqlite3
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

# with sqlite3.connect('allweather_portfolio.db') as db:
#   prices = pd.read_sql('SELECT * from [All_Weather_Portfolio]', db)  

# BIZDAYS_A_YEAR = (end-start).days / years #This is wrong
BIZDAYS_A_YEAR = 252
daily_ret = prices.pct_change().add(1).cumprod()
annual_ret = np.power(daily_ret[-1:], 1/years) - 1
daily_cov = prices.pct_change().cov()
annual_cov = daily_cov * BIZDAYS_A_YEAR

# sns.set()
# cormat = prices.corr()
# fig, ax = plt.subplots(1,2)
# pairs = sns.PairGrid(prices)
# pairs.map(sns.scatterplot)
# sns.heatmap(cormat, cmap='YlGnBu', annot=True, linewidths=0.1, ax=ax[1])
# plt.show()

cormat = prices.corr()

sns.set(font_scale=0.5)
sns.PairGrid(prices).map(sns.scatterplot)
plt.show()

sns.set()
fig1, ax1 = plt.subplots(1)
sns.heatmap(cormat, cmap='YlGnBu', annot=True, linewidths=0.1, ax=ax1)
plt.show()
