import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir(r'D:\myprojects')

start = datetime(2010, 7, 23)
end = datetime.today()
years = tuple(range(start.year, end.year+1))

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

# for date in prices.index:
#     prices.loc[date, 'DayofYear'] = pd.Period(date, freq='D').day_of_year

weights = {
    'SPY' : [12],
    'EFA' : [12],
    'EEM' : [12],
    'DBC' : [7],
    'GLD' : [7],
    'EDV' : [18],
    'LTPZ' : [18],
    'LQD' : [7],
    'EMLC' : [7]
}    
weights = pd.DataFrame(weights)
weights.index = [pd.Timestamp(start)]

days = {'first':[], 'last':[]}
for year in range(start.year, end.year+1):
    nday = prices.loc[prices.index.year==year, 'DayofYear']  
    days['first'].append(nday.loc[nday==nday.min()].index)
    days['last'].append(nday.loc[nday==nday.max()].index)
days = pd.DataFrame(days, index=range(start.year, end.year+1))
days['days'] = days['last'] - days['first']

daily_ret = prices.pct_change()

yearly_cumprod = pd.DataFrame()
for year in years:
    yearly_cumprod = pd.concat([yearly_cumprod, daily_ret.loc[daily_ret.index.year==year].add(1).cumprod()])

yearly_ret = pd.DataFrame()
for year in years:
    returns = np.power(yearly_cumprod.loc[yearly_cumprod.index.year==year].iloc[-1,:], 1/days['days'][year][0].days) - 1
    yearly_ret = pd.concat([yearly_ret, returns], axis=1)
# yearly_ret = yearly_ret[0:len(yearly_ret)-1] # not used because the above 'DayofYear' column is not created
yearly_ret = yearly_ret.T
yearly_ret = pd.concat([yearly_ret, pd.Series(yearly_ret.sum(axis=1), name='Total')], axis=1)

all_year_cumprod = daily_ret.add(1).cumprod()
# all_year_cumprod = all_year_cumprod.loc[:, all_year_cumprod.columns!='DayofYear'] # not used because the above 'DayofYear' column is not created
all_year_ret = np.power(all_year_cumprod.iloc[-1:], 1/days['days'].sum()[0].days) - 1
all_year_total_ret = all_year_ret.sum(axis=1)
all_year_total_ret = all_year_total_ret[-1]

all_year_daily_cov = prices.pct_change().cov()
yearly_daily_cov = pd.DataFrame()
for year in years:
    yearly_daily_cov = pd.concat([yearly_daily_cov, daily_ret[daily_ret.index.year==year].cov()]) 

yearly_asset_stds = pd.DataFrame()
for year in years:
    yearly_asset_stds = pd.concat([yearly_asset_stds, prices.loc[prices.index.year==year].std()], axis=1)
yearly_asset_stds = yearly_asset_stds.T
yearly_asset_stds.index = years
yearly_max_stds = yearly_asset_stds.max()
# yearly_max_stds = yearly_max_stds[:len(yearly_asset_stds.max())-1] # not used because the above 'DayofYear' column is not 

yearly_total_stds = []
pricesums = prices.sum(axis='columns')
for year in years:
    yearly_total_stds.append(pricesums.loc[pricesums.index.year==year].std())
yearly_total_stds = pd.DataFrame(yearly_total_stds, index = years)
yearly_total_stds.columns = ['']

all_year_class_stds = prices.std()
all_year_total_stds = prices.sum(axis='columns').std()

# all_year_class_stds = all_year_class_stds[:len(prices.std())-1] # not used because the above 'DayofYear' column is not created

yearly_peaks = pd.DataFrame()
for year in years:
    yearly_peaks = pd.concat([yearly_peaks, prices.loc[prices.index.year==year].max()], axis=1)
yearly_peaks = yearly_peaks.T
yearly_peaks.index = years

all_year_peaks = prices.max()

yearly_drawdowns = pd.DataFrame()
for year in years:
    yearly_drawdowns = pd.concat([yearly_drawdowns, prices.loc[prices.index.year==year]/yearly_peaks.loc[year] - 1])

yearly_mdds = pd.DataFrame()
for year in years:
    yearly_mdds = pd.concat([yearly_mdds, yearly_drawdowns.loc[yearly_drawdowns.index.year==year].min()], axis=1) #same as axis='column'
yearly_mdds.columns = years
yearly_mdds = yearly_mdds.T

all_year_drawdowns = prices/all_year_peaks - 1
all_year_mdds = all_year_drawdowns.min()
# all_year_mdds = all_year_mdds[:len(all_year_drawdowns.min())-1] # not used because the above 'DayofYear' column is not created


risk_free = yearly_ret.sub(yearly_ret['Tresuary Inflation-Protected Securities'], axis='index')
yearly_asset_stds.index = risk_free.index
yearly_total_stds.index = risk_free.index
yearly_sharpe = pd.concat([risk_free.loc[:,risk_free.columns!='Total'].div(yearly_asset_stds), \
    pd.Series(risk_free['Total']/yearly_total_stds[''], name='Total')], axis='columns')


all_year_class_sharpe = (all_year_ret-all_year_ret['Tresuary Inflation-Protected Securities'][0])/all_year_class_stds
all_year_total_sharpe = (all_year_ret.sum(axis='columns')[0]-all_year_ret['Tresuary Inflation-Protected Securities'][0])/all_year_total_stds


# print(f'CAGR: {cagr}, Max Risk: {max_risk}, Sharpe Ratio: {sharpe}, MDD: {mdd}')
