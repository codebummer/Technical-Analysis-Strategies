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
# because when multiplying, index names (or column names) are often different, 
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
    
    # The rebalancing algorithm is as follow:
    # 1. find off values = calculate prices lower or higher than allocation plans
    #   off values = yearend prices - yearend prices * allocation ratio
    off_values = values.loc[yearend] - weights/100*values.loc[yearend].sum()
    
    # 2. find off quantities = calculate asset quantities lower or higher than allocation plans, based on off values
    #   off quantities = off values / year end prices
    #   round the above off quantities
    #   convert them to integer
    off_qty = (off_values/yearly_prices.loc[yearend]).round(decimals=1).astype('int')   
    
    
    # 3. find gains = calculate money amounts of assets higher than allocation plans, based on off quantities
    #   make all minus off quantities 0 = use pandas.clip(lower=0)
    #   dataframe.clip(lower=0) will keep all elements and prevent NaNs for matrix operations
    #   dataframe.loc[year,off_qty<0] will remove all elements that do not match the condition, 
    #   which will result NaNs and NaNs will cause errors for matrix operations.
    #   The following is an example of the bad practice described above
    #   gains = off_qty.loc[off_qty>0] * yearly_prices.loc[yearend,off_qty>0] 
    gains = off_qty.clip(lower=0) * yearly_prices.loc[yearend]
    
    # 4. find losses = calculate money amounts of assets lower than allocation plans, based on off quantities
    #   same as gains
    #   The following is an example of the bad practice 
    #   losses = off_qty.loc[off_qty<0] * yearly_prices.loc[yearend,off_qty<0]
    losses = off_qty.clip(upper=0) * yearly_prices.loc[yearend]
    
    # 5. if the total gains are less than a minimum unit price of the loss assets,
    #   skip rebalancing, because not a single minimum asset can be purchased
    #
    #   a. To find the minimum unit price of the loss assets,
    #      find unit prices of loss assets
    #
    #   b. convert nonzero quantities of loss assets to 1: 
    #      off_qty.clip(upper=0).where(off_qty.clip(upper=0)==0,1)
    #   
    #   c. multiply them by yearend prices, which results in loss unit prices
    loss_unit_prices = off_qty.clip(upper=0).where(off_qty.clip(upper=0)==0,1) * yearly_prices.loc[yearend]
    
    #   d. exclude zeros to accurately locate minimum unit price
    #      loss_unit_prices.loc[loss_unit_prices!=0]
    
    #   e. sort loss unit prices that are not zero.
    loss_unit_prices = loss_unit_prices.loc[loss_unit_prices!=0].sort_values()
    #   f. locate the minimum unit price of loss assets
    min_asset, min_unit_price = loss_unit_prices.index[0], loss_unit_prices[0]    
    #   g. execute with the above conditions to skip the rebalancing
    if gains.sum() < min_unit_price:
        continue
    
    # The following is rather inaccurate condition to skip rebalancing
    # because it looks at the minimum amount of the total loss asset,
    # not the minimum amount of the unit loss asset
    # if gains.sum() < losses.min(): 
    #     continue
    

    # 6. find asset quantities to restock
    # quantify asset quantities to restock from money amounts to restock
    #   a. find rebalance ratio 
    #       = find ratio of loss assets' off quantities
    #       off quantities of only loss assets / sum of off quantities of only loss assets   
    rebalance_ratio = off_qty.clip(upper=0) / off_qty.clip(upper=0).sum()
    #   b. find amounts ratio that can be distributed to buy restocking assets after selling gained assets
    #       sum of gains * off quantities ratio of loss assets / yearend prices
    #   c. to find quantities to restock,
    #       round the restock quantitiy ratio above that can be distributed from selling gained assets
    rebalance_qty = (gains.sum()*rebalance_ratio/yearly_prices.loc[yearend]).round(decimals=1).astype('int')
    #   d. find actual prices to restock
    #       quantities to restock * yearend prices
    #   e. sort the restock prices calculated
    rebalance_order = (rebalance_qty.abs()*yearly_prices.loc[yearend]).sort_values(ascending=False)
    #   f. find the threshold to buy restocking assets
    #       calculate cumulative sum of the above sorted restock prices
    #       list restock prices of assets that are less than the total gains
    rebalance_assets = rebalance_order.cumsum() < gains.sum() 

    # 7. correct holdings accoringly
    #   rebalance_qty.loc[rebalance_assets] refers to assets to restock (plus values so add this from holdings)
    #   off_qty.loc[off_qty>0] refers to assets to sell (plus values so subract this from holdings)
    holdings.loc[yearend] = holdings.loc[yearend] + rebalance_qty.loc[rebalance_assets] - off_qty.clip(lower=0)

FX_included_prices = pd.DataFrame()
for year in years:
    FX_included_prices = pd.concat([FX_included_prices, (prices.loc[prices.index.year==year]).multiply(krw.loc[krw.index.year==year,'cumprod'], axis='index')])
FX_included_prices


portfolio_prices = {'FX_included': pd.DataFrame(), 'FX_not_included': pd.DataFrame()}
for year in years:
    portfolio_prices['FX_not_included'] = pd.concat([portfolio_prices['FX_not_included'], prices.loc[prices.index.year==year]*holdings.loc[holdings.index.year==year].iloc[0,:]])
    portfolio_prices['FX_included'] = pd.concat([portfolio_prices['FX_included'], FX_included_prices.loc[FX_included_prices.index.year==year]*holdings.loc[holdings.index.year==year].iloc[0,:]])
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
plt.legend(portfolio_prices['FX_not_included'].columns, loc='best')
plt.show()

port = holdings*yearly_prices
allocation_diff = (port.divide(port.sum(axis='columns'), axis='index') - weights/weights.sum()) * 100
allocation_diff.loc[allocation_diff.values>1][allocation_diff>1]


# The below should be rechecked and rewritten if necessary


days = {'first':[], 'last':[]}
for year in range(start.year, end.year+1):
    nday = prices.loc[prices.index.year==year, 'DayofYear']  
    days['first'].append(nday.loc[nday==nday.min()].index)
    days['last'].append(nday.loc[nday==nday.max()].index)
days = pd.DataFrame(days, index=range(start.year, end.year+1))
days['days'] = days['last'] - days['first']

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
