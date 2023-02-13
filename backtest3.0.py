import pandas as pd
import numpy as np
from benchmark_datareader import Benchmark
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

'''
For backtesting, make the price matrix and the holdings matrix.
The price matrix is the portfolio assets' prices.
The holdings matrix is determined by rebalancing

asset values = price matrix * holdings matrix
backtest the asset values
'''

# initial amount of investment
invested = 30_000
# weights for assets in ratio
weights = pd.Series({'S&P500':0.3, 'US10Y':0.5, 'XAU/USD':0.15, 'USD':0.05}, name='Weights')

benchmark = Benchmark()
assets = benchmark.get()
# benchmark.plot_returns(assets,[datetime(2018,1,2),datetime(2021,12,31)])
bonds, nonbonds = benchmark.columns_except(['US10Y'], assets)
cash, noncash = benchmark.columns_except(['USD'], assets)
bonds = bonds[0]
cash = cash[0]
assets[bonds] = benchmark.yields_to_prices(10, assets[bonds], False)

# initial amount of investment
invested = 30_000
# weights for assets in ratio
weights = pd.Series({'S&P500':0.3, 'US10Y':0.5, 'XAU/USD':0.15, 'USD':0.05}, name='Weights')

holdings = pd.Series({'S&P500':30, 'US10Y':50, 'XAU/USD':15, 'USD':5}, name='Holdings')

def find_periods(df):
    '''
    gets a dataframe, indexed with datetime.timestamp, as an input, 
    returns a list that contains tuple pairs of start and end dates for each year in the dataframe        
    df: a dataframe of daily prices of assets, indexed with datetime.timestamp
    '''
    periods = []
    for year in tqdm(list(set(df.index.year))):
        dates = df.groupby(df.index.year).get_group(year).index
        periods.append((dates[0],dates[-1]))        
    return periods

# make weight matrix
def make_weight_matrix(period, holdings, assets):
    '''
    period: a tuple pair of start and end datetime.timestamp
    holdings: holdings ratio by which the weight matrix is generated
    assets: a dataframe which contains asset daily prices with date index from which the weight matrix will copy its date index
    returns a dataframe which is a weight matrix for the given period
    '''
    start, end = period
    dates = assets.loc[start:end].index
    return pd.DataFrame([holdings.values for _ in  dates], index=dates, columns=holdings.index)    

# make holdings matrix by rebalancing holdings
def make_holdings_matrix(weights, holdings, periods, assets):
    '''
    weights: assets' weights you planned for the strategy (pandas.Series)
    holdings: unit numbers of assets(i.e. stock numers for stocks) you hold in the begining (pandas.Series)
    periods: [(datetime(start date), datetime(end date)), (datetime(start date), datetime(end date))...] 
             periods can be easily generated by Benchmark().find_periods() method
    assets: dataframe of daily assets prices
    
    returns holdings matrix, rebalancing every period handed over
    
    '''
    holdings_matrix = pd.DataFrame()

    for start, end in tqdm(periods):
        prior_holdings = holdings
        while True:
            values = assets.loc[end] * holdings
            off_values = values - values.sum()*weights  #off_values = off_remainders*assets.loc[end]
            off_real_qty = off_values / assets.loc[end]
            off_quotients = off_real_qty.astype('int64')
            # off_remainders = off_real_qty - off_quotients  
            holdings -= off_quotients     
            if off_quotients.sum() == 0:
                break
            

        print('\noff-values:\n', off_values, '\noff-values/asset prices: \n', off_values/assets.loc[end])
        if (prior_holdings*assets.loc[end]).sum() != (holdings*assets.loc[end]).sum():
            print('rebalaced amount is not equal to prior balance')
            break
        
        holdings_matrix = pd.concat([holdings_matrix, make_weight_matrix((start,end), holdings, assets)])    
    
    return holdings_matrix

def make_returns_matrix(periods, values_matrix, returns_periods='annualcum'):
    '''
    periods: Benchmark().find_periods generated start/end datetime pairs that indicate rebalancing periods
    values_matrix: price matrix * holdings matrix (holdings matrix includes weights, so don't factor in weights again)
    returns_periods: periods by which returns are calculated
        "annualcum" generates entire period annual returns cumulated
        "annual" generates annual returns just yearly, not cumulated
        "daily" generates daily returns    
    
    returns both cumprods matrix and returns matrix 
    if you want just returns, not returns matrix, use Benchmark().returns_matrix_to_returns()
    '''
    cumprods = pd.DataFrame()
    returns = pd.DataFrame()

    if returns_periods == 'annualcum':
        years = {}
        for year, num in zip(set(values_matrix.index.year), range(1,len(values_matrix.index.year)+1)):
            years[year] = num

        cumprods = values_matrix.pct_change().add(1).cumprod()
        for start, end in tqdm(periods):            
            returns = pd.concat([returns, cumprods.multiply(1/years[end.year], axis='index')]) 
    elif returns_periods == 'annual':    
        for start, end in tqdm(periods):
            cumprods = pd.concat([cumprods, values_matrix.loc[start:end].pct_change().add(1).cumprod()])
            returns = pd.concat([returns, cumprods])
    elif returns_periods == 'daily':
        for start, end in tqdm(periods):
            cumprods = pd.concat([cumprods, values_matrix.loc[start:end].pct_change().add(1).cumprod()])
            exp = pd.Series(cumprods.index.map(lambda x:1/x.timetuple().tm_yday), index=cumprods.index, name='1/Days')
            returns = pd.concat([returns, cumprods.multiply(exp, axis='index')])

    return cumprods, returns


def returns_matrix_to_returns(periods, returns_matrix):
    '''
    periods: Benchmark().find_periods generated start/end datetime pairs that indicate rebalancing periods
    returns_matrix: Benchmark().make_returns_matrix() generated dataframe
    
    returns both each asset's respective returns and all assets' returns
    '''
    returns = pd.DataFrame()
    for start, end in tqdm(periods):
        returns = pd.concat([returns, returns_matrix.loc[end]])
    return returns.sum(), returns.sum(axis='columns')



periods = find_periods(assets)
holdings_matrix = make_holdings_matrix(weights, holdings, periods, assets)

values_matrix = holdings_matrix*assets
cumprods_matirx, returns_matrix = make_returns_matrix(periods, values_matrix)
asset_returns, total_returns = returns_matrix_to_returns(periods, returns_matrix)

allcum = all.pct_change().add(1).cumprod()
returns = all.sum(axis='columns')
returnscum = returns.pct_change().add(1).cumprod()
returnscum.columns = ['All']
all.groupby(all.index.year).get_group(1980)
sns.lineplot(returns)
sns.lineplot(all.loc[datetime(1980,1,2):])
sns.lineplot(returnscum[datetime(1980,1,2):])
sns.lineplot(allcum.loc[datetime(1980,1,2):])
sns.barplot(allcum.loc[datetime(1980,1,2):])

sns.heatmap(assets.corr(), annot=True)
plt.show()
