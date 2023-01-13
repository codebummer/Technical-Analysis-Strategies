import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime
# from Investar import Analyzer


start = datetime(2019, 1, 1)
end = datetime(2022, 11, 4)

def fetch_prices(stocks):
    df = pd.DataFrame([])
    for ticker, stock in stocks.items():
         df[stock] = (web.DataReader(ticker, 'yahoo', start, end)['Adj Close'])
    return df

stocks = {'005930.KS' : 'Samsung', 
        #   '900310.KQ' :' Coloray', 
          '005490.KS' : 'POSCO Holdings',
        #   '012700.KQ' : 'Leaqdcorp',
          '017670.KS' : 'SKTelecom', 
          '033780.KS' : 'KT&G'}


df = fetch_prices(stocks)

BIZDAYS_A_YEAR = 252
daily_ret = df.pct_change()
annual_ret = daily_ret.mean() * BIZDAYS_A_YEAR
daily_cov = daily_ret.cov()
annual_cov = daily_cov * BIZDAYS_A_YEAR

# daily_ret = df.pct_change().add(1).cumprod().subtract(1)
# annual_ret = daily_ret[BIZDAYS_A_YEAR]
# daily_cov = df.pct_change().cov()


port_ret, port_risk, port_weights = [], [], []

for _ in range(20_000):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    
    returns = np.dot(weights, annual_ret)
    risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
    # print(weights.T)
    
    port_ret.append(returns)
    port_risk.append(risk)
    port_weights.append(weights)

portfolio = {'Returns' : port_ret, 'Risk' : port_risk}
for i, s in enumerate(stocks.values()):
    portfolio[s] = [weight[i] for weight in port_weights]

df = pd.DataFrame(portfolio)
df = df[['Returns', 'Risk'] + [s for s in stocks.values()]]

df.plot.scatter(x='Risk', y='Returns', figsize=(8,6), grid=True)
plt.title('Efficient Frontier')
plt.xlabel('Risk')
plt.ylabel('Expected Returns')
plt.show()

    


