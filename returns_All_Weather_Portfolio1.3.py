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
        # df[stock] = (web.get_data_yahoo(ticker, start, end)['Close'])
        df[ticker] = (web.get_data_yahoo(ticker, start, end)['Close'])
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
    'EMLC' : 'Emerging Market Bonds',
    # 'KRW=X' : 'USD/KRW'
}

prices = fetch_prices(stocks)

krw = fetch_prices({'KRW=X':'USD/KRW'})
krw['change'] = krw.pct_change()
krw_cumprod = pd.DataFrame()
for year in years:
    krw_cumprod = pd.concat([krw_cumprod, krw.loc[krw.index.year==year,'change'].add(1).cumprod()])
krw = pd.concat([krw,krw_cumprod], axis='columns')
krw.columns = list(krw.columns)[:-1]+['cumprod']

with sqlite3.connect('allweather_portfolio.db') as db:
    prices.to_sql('Stock_Prices', db, if_exists='replace')

# for date in prices.index:
#     prices.loc[date, 'DayofYear'] = pd.Period(date, freq='D').day_of_year

weights = {
    'SPY' : 12,
    'EFA' : 12,
    'EEM' : 12,
    'DBC' : 7,
    'GLD' : 7,
    'EDV' : 18,
    'LTPZ' : 18,
    'LQD' : 7,
    'EMLC' : 7
}   
# create a series instead of a dataframe, 
# because when multiplying, indices are often different, 
# which causes incorrect results 
weights = pd.Series(weights) 

invested = 10_000

# use the following statement, when weights is defined as a dataframe. 
# Because the indices are different, 
# multiplying, indices should be removed
# weights.iloc[0,:] * prices.head() 


# The following immediate block is just for initial plotting purposes.
weighted_prices = weights*prices
daily_amounts = weighted_prices.sum(axis='columns')
sns.set()
fig, ax = plt.subplots(1,2)
ax[0].plot(daily_amounts)
ax[1].plot(weighted_prices)
plt.legend(weighted_prices.columns)
plt.show()

holdings = invested * weights / prices.iloc[0,:]
holdings = holdings.astype('int')
holdings = holdings.to_frame(name=pd.Timestamp(start))
holdings = holdings.T

values = holdings * prices.iloc[0,:]

yearly_prices = pd.DataFrame(prices.iloc[0,:], columns=[prices.index[0]])
for year in years:
    yearly_prices = pd.concat([yearly_prices, prices.loc[prices.index.year==year].iloc[-1,:]], axis='columns')
yearly_prices = yearly_prices.T

daily_ret = prices.pct_change()
yearly_cumprod = pd.DataFrame()
for year in years:
    # yearly_cumprod = pd.concat([yearly_cumprod, daily_ret.loc[daily_ret.index.year==year].add(1).cumprod()])
    yearly_cumprod = pd.concat([yearly_cumprod, daily_ret.loc[daily_ret.index.year==year].add(1).cumprod().iloc[-1,:]], axis='columns')
yearly_cumprod = yearly_cumprod.T

# rebalancing portfolio assets according to asset allocation plans, or 'weights' in this code
for yearend in yearly_prices.index:    
    if yearend == yearly_prices.index[0]:
        continue
    add = pd.DataFrame(yearly_prices.loc[yearend] * holdings.iloc[-1,:], columns=[yearend]).T
    values = pd.concat([values, add])
    
    add = values.loc[yearend]/yearly_prices.loc[yearend]
    holdings = pd.concat([holdings, add.round(decimals=1).astype('int').to_frame().T])

    off_values = values.loc[yearend] - weights/100*values.loc[yearend].sum()
    off_qty = (off_values/yearly_prices.loc[yearend]).round(decimals=1).astype('int')   
    
    gains = off_qty.clip(lower=0) * yearly_prices.loc[yearend]
    losses = off_qty.clip(upper=0) * yearly_prices.loc[yearend]
    # changes elements, which results NaN in later operations
    # use clip() so as to keep elements intact, which results no NaN in later operations
    # so, the following statements are not recommended
    # gains = off_qty.loc[off_qty>0] * yearly_prices.loc[yearend,off_qty>0] 
    # losses = off_qty.loc[off_qty<0] * yearly_prices.loc[yearend,off_qty<0]
    
    loss_unit_prices = off_qty.clip(upper=0).where(off_qty.clip(upper=0)==0,1) * yearly_prices.loc[yearend]
    loss_unit_prices = loss_unit_prices.loc[loss_unit_prices!=0].sort_values()
    min_asset, min_unit_price = loss_unit_prices.index[0], loss_unit_prices[0]    
    if gains.sum() < min_unit_price:
        continue
    
    # The following is rather inaccurate condition to skip rebalancing
    # because it looks at the minimum amount of the total loss asset,
    # not the minimum amount of the unit loss asset
    # if gains.sum() < losses.min(): 
    #     continue
    
    # quantify asset quantities to restock from money amounts to restock
    rebalance_ratio = off_qty.clip(upper=0) / off_qty.clip(upper=0).sum()
    rebalance_qty = (gains.sum()*rebalance_ratio/yearly_prices.loc[yearend]).round(decimals=1).astype('int')
    rebalance_order = (rebalance_qty.abs()*yearly_prices.loc[yearend]).sort_values(ascending=False)
    rebalance_assets = rebalance_order.cumsum() < gains.sum() 

    # correct holdings accoringly
    # rebalance_qty.loc[rebalance_assets] refers to assets to restock (plus values so add this from holdings)
    # off_qty.loc[off_qty>0] refers to assets to sell (plus values so subract this from holdings)
    holdings.loc[yearend] = holdings.loc[yearend] + rebalance_qty.loc[rebalance_assets] - off_qty.clip(lower=0)

FX_included_prices = pd.DataFrame()
for year in years:
    FX_included_prices = pd.concat([FX_included_prices, (prices.loc[prices.index.year==year]).multiply(krw.loc[krw.index.year==year,'cumprod'], axis='index')])
FX_included_prices


portfolio_prices = {'FX_included': pd.DataFrame(), 'FX_not_included': pd.DataFrame()}
for year in years:
    portfolio_prices['FX_not_included'] = pd.concat([portfolio_prices['FX_not_included'], prices.loc[prices.index.year==year]*holdings.loc[yearend]])
    portfolio_prices['FX_included'] = pd.concat([portfolio_prices['FX_included'], FX_included_prices.loc[FX_included_prices.index.year==year]*holdings.loc[yearend]])
portfolio_daily_values = {'FX_not_included':portfolio_prices['FX_not_included'].sum(axis='columns'),
                          'FX_included':portfolio_prices['FX_included'].sum(axis='columns')}

portfolio_daily_values['FX_included'].loc[portfolio_daily_values['FX_included'].isna()]
any(portfolio_daily_values['FX_included'].isna())

portfolio_daily_values['FX_included'] = portfolio_daily_values['FX_included'].loc[portfolio_daily_values['FX_included']!=0]

sns.set()
fig1, ax1 = plt.subplots(2,2)
ax1[0,0].plot(portfolio_daily_values['FX_not_included'])
ax1[0,1].plot(portfolio_prices['FX_not_included'])
ax1[1,0].plot(portfolio_daily_values['FX_included'])
ax1[1,1].plot(portfolio_prices['FX_included'])
plt.legend(portfolio_prices['FX_not_included'].columns)
plt.show()

port = holdings*yearly_prices
allocation_diff = (port.divide(port.sum(axis='columns'), axis='index') - weights/weights.sum()) * 100
allocation_diff.loc[allocation_diff.values>1][allocation_diff>1]


daily_ret = prices.pct_change()

yearly_cumprod = pd.DataFrame()
for year in years:
    # yearly_cumprod = pd.concat([yearly_cumprod, daily_ret.loc[daily_ret.index.year==year].add(1).cumprod()])
    yearly_cumprod = pd.concat([yearly_cumprod, daily_ret.loc[daily_ret.index.year==year].add(1).cumprod().iloc[-1,:]], axis='columns')
yearly_cumprod = yearly_cumprod.T

for year in years:
    yearly_cumprod.loc[yearly_cumprod.index.year==year]
(yearly_cumprod*weights)
annual = yearly_cumprod.loc[yearly_cumprod.index.year==2010]
off_balance = annual.iloc[0,:]/annual.sum(axis='columns')[0]-weights/100
off_balance.sort_values()

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
