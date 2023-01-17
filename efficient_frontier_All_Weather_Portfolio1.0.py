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

port_ret, port_risk, port_weights, sharpe_ratio = [], [], [], []

for _ in range(200_000):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    
    returns = np.dot(weights, annual_ret.iloc[0])
    risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
    # risk = np.sqrt(np.multiply(weights.T, np.dot(annual_cov, weights))) # same as above
    # risk = np.sqrt(weights.T * np.dot(annual_cov, weights))) # same as above
    
    port_ret.append(returns)
    port_risk.append(risk)
    port_weights.append(weights)
    sharpe_ratio.append(returns/risk)

portfolio = {'Returns' : port_ret, 'Risk' : port_risk, 'Sharpe' : sharpe_ratio}
for i, s in enumerate(stocks.values()):
    portfolio[s] = [weight[i] for weight in port_weights]
  
df = pd.DataFrame(portfolio)
df = df[['Returns', 'Risk', 'Sharpe'] + [s for s in stocks.values()]]

max_sharpe = df.loc[df['Sharpe']==df['Sharpe'].max()]
min_risk = df.loc[df['Risk']==df['Risk'].min()]

with sqlite3.connect('allweather_portfolio.db') as db:
  max_sharpe.to_sql('Sharpe_Ratio_Maximized_Portfolio', db, if_exists='replace')
  min_risk.to_sql('Risks_Minimized_Portfolio', db, if_exists='replace')
  
print(f'Sharpe Ratio Maximized: {max_sharpe} \nRisks Minmized: {min_risk}')

sns.set()
fig, ax = plt.subplots(3,3, layout='constrained')
count = 0
for i in range(3):
  for j in range(3):
    sns.distplot(prices.iloc[:,count], bins=25, color='g', ax=ax[i,j])
    count += 1
plt.show() 

fig, ax = plt.sutplots()
sns.displot(prices, kind='kde')
plt.show()

df.plot.scatter(x='Risk', y='Returns', c='Sharpe', cmap='viridis', edgecolors='k', figsize=(11,7), grid=True)
plt.scatter(x=max_sharpe['Risk'], y=max_sharpe['Returns'], c='r', marker='*', s=300)
plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='r', marker='X', s=200)
plt.title('Efficient Frontier / Portfolio Optimization')
plt.xlabel('Risk')
plt.ylabel('Expected Returns')
plt.show()
