import pandas as pd
import pandas_datareader.data as web
import investpy
import yfinance as yf
import numpy as np
import numpy_financial as npf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir(r'D:\myprojects')

start = datetime(1962, 1, 1)
end = datetime.today()
years = tuple(range(start.year, end.year+1))

yf.pdr_override()
def fetch_prices(stocks):
    df = pd.DataFrame([])
    for ticker, stock in stocks.items():
        df[ticker] = (web.get_data_yahoo(ticker, start, end)['Close'])
    return df

stocks = {
    '^GSPC' : 'S&P 500',
    '^TNX' : 'US Tresuary 10 Years'
}

prices = fetch_prices(stocks)

us10y = investpy.get_bond_historical_data('US10YT=X', start.strftime('%d/%m/%Y'), end.strftime('%d/%m/%Y'), as_json=True)

weights = {
    '^GSPC' : 70,
    '^TNX' : 30
}   

weights = pd.Series(weights) 
invested = 10_000

prices = pd.concat([prices, pd.Series([prices.iloc[0,0] for _ in range(len(prices))],index=prices.index)], axis='columns')
prices.columns = ['GSPC','TNX','first']
prices['returns'] = prices['GSPC']/prices['first'] - 1
returns = prices[['returns', 'TNX']]
returns.columns = ['GSPC', 'TNX']
sums = returns.sum(axis='columns')



restocks = pd.DataFrame()
for year in tuple(range(start.year, end.year, 10)):
    # idx = prices.index[prices.index.year==year] + pd.to_timedelta(365*10,unit='D')
    reprice = prices.loc[prices.index.year==year]
    reprice = reprice.loc[reprice.index.month==start.month].iloc[0,:]
    restocks = pd.concat([restocks, reprice], axis='columns')
restocks = restocks.T

decades = [prices.index[0]]
for _ in range(int(len(years)/10)+1):
    decades.append(decades[-1] + pd.to_timedelta(3650, unit='D'))

tens = []
for idx, decade in enumerate(decades):
    tens.append(prices.index[prices.index<=decades[idx]][-1])

maturities = pd.DataFrame()
for idx in range(len(tens)-1):
    screened = prices.index[prices.index<=tens[idx+1]]
    screened = screened[screened>=tens[idx]]
    maturities = pd.concat([maturities, pd.DataFrame([tens[idx+1] for _ in range(len(screened))], index=screened, columns=['maturities'])])

prices = pd.concat([prices, maturities], axis='columns')
prices['days'] = prices['maturities'] - prices.index

prices.index + pd.Series([relativedelta(years=10) for _ in range(len(prices))], index=prices.index)

bond_list = investpy.get_bonds_list()
bond_list = pd.Series(bond_list)
bond_list.loc[['U.S.' in _ for _ in bond_list.values]]

investpy.get_bond_historical_data('U.S. 10Y', start.strftime('%d/%m/%Y'), end.strftime('%d/%m/%Y'))
investpy.get_bonds().groupby('country').apply(lambda x:x)
investpy.get_bonds().loc[investpy.get_bonds().country.str.contains('united states')]

indices = investpy.get_indices()
indices.loc[indices.name.str.contains('S&P 500')]
investpy.get_index_historical_data('S&P 500', 'united states', start.strftime('%d/%m/%Y'), end.strftime('%d/%m/%Y'))


prices.index.to_period('M') - prices.maturities.to_period('M')
prices.maturities.values.apply(pd.to_period('M'))


np.power(1_000, 1/prices['days'].dt.days) + prices['TNX']*1_000

prices.days[0].days

bonds = pd.DataFrame()
for year in years:
    to_prices = npf.pv(prices['TNX'], prices['days'], prices['TNX'][0], 1_000)



bond_price = (npf.pv(annual_interest, period, coupon_payment, principal)) * -1
npf.pv(prices.TNX[0], )

sns.set()
fig, ax = plt.subplots(1)
ax.plot(sums)
ax.set_title('S&P500 and US Treasury 10 Years')
plt.show()
