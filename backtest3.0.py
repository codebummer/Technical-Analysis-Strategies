import pandas as pd
import numpy as np
from benchmark_datareader import Benchmark
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt
import re

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'


'''
For backtesting, make the price matrix and the holdings matrix.
The price matrix is the portfolio assets' prices.
The holdings matrix is determined by rebalancing

asset values = price matrix * holdings matrix
backtest the asset values
'''

benchmark = Benchmark()
assets = benchmark.get()
eikon = benchmark.load_eikon()
french = benchmark.load_french()
# benchmark.plot_returns(assets,[datetime(2018,1,2),datetime(2021,12,31)])
bonds, nonbonds = benchmark.columns_except(['US10Y'], assets)
cash, noncash = benchmark.columns_except(['USD'], assets)
bonds = bonds[0]
cash = cash[0]
assets[bonds] = benchmark.yields_to_prices(10, assets[bonds], False)

# initial amount of investment
invested = 1_000
# weights for assets in ratio
weights = pd.Series({'S&P500':0.3, 'US10Y':0.5, 'XAU/USD':0.15, 'USD':0.05}, name='Weights')

lasts = [end for start, end in tqdm(periods) for _ in range(len(assets.groupby(assets.index.year).get_group(end.year)))]
assets['Signals'] = np.where(assets.index==lasts,1,0)
positions = assets['Signals'].cumsum().replace(to_replace=0, method='ffill')
positions.groupby(positions.index.year).get_group(1874)

def load_french(period='monthly'):   
    '''
    period: "daily" or "monthly" for daily or monthly prices dataframe
    returns two dataframes
    ''' 
    if period == 'daily':        
        url_daily = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_25_Portfolios_ME_BE-ME_daily_CSV.zip'
        data = pd.read_csv(url_daily, index_col=0, header=11, parse_dates=True)            
    elif period == 'monthly':
        url_monthly = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_25_Portfolios_ME_BE-ME_CSV.zip'
        data = pd.read_csv(url_monthly, index_col=0, header=12, parse_dates=True)
        
    def _generate_index(index):
        index = index.map(lambda x:x.strip())
        dates = []
        breaks = [0]
        for idx in tqdm(range(len(index))):
            string = re.search('[a-zA-Z+]', index[idx])
            if index[idx] == '':
                dates.append('')
            elif string:
                dates.append(index[idx])
                breaks.append(idx)
            else:
                if len(index[idx]) == 8:
                    dates.append(datetime.strptime(index[idx],'%Y%m%d'))
                if len(index[idx]) == 6:               
                    dates.append(datetime.strptime(index[idx],'%Y%m'))
                elif len(index[idx]) == 4:
                    dates.append(datetime.strptime(index[idx],'%Y'))
        breaks.append(len(data))
        return np.array(dates), breaks

    data.index, breaks = _generate_index(data.index)

    results = {}
    for cut in tqdm(range(len(breaks)-1)):
        if cut == 0:
            results['Average Value Weighted Returns -- Monthly'] = data.iloc[:breaks[cut+1]]
        else:    
            results[data.index[breaks[cut]]] = data.iloc[breaks[cut]:breaks[cut+1]]
            
    return results

def initial_holdings(invested, weights, assets, cash):
    '''
    invested: initial investing cash amount - int or float
    weights: weights of investment for each asset - pandas.Series
    assets: dataframe of assets' daily prices
    cash: column name of the "cash" asset in the assets - 'str' not a list
    returns a pandas.Series that includes inital numbers of holdings for each asset
    '''
    quotients = invested*weights//assets.iloc[0,:]
    remainders = invested*weights/assets.iloc[0,:] - quotients
    quotients[cash] += (remainders*assets.iloc[0,:]).sum()
    quotients.name = 'Holdings'
    if (quotients*assets.iloc[0,:]).sum() == invested:
        return quotients
    else:
        print('The holdings amount does not match the invested amount')
    
holdings = initial_holdings(invested, weights, assets, 'USD')

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
            returns = pd.concat([returns, cumprods.loc[start:end].pow(1/years[end.year], axis='index')]) 
    elif returns_periods == 'annual':    
        for start, end in tqdm(periods):
            cumprods = pd.concat([cumprods, values_matrix.loc[start:end].pct_change().add(1).cumprod()])
            returns = pd.concat([returns, cumprods.loc[start:end]])
    elif returns_periods == 'daily':
        for start, end in tqdm(periods):
            cumprods = pd.concat([cumprods, values_matrix.loc[start:end].pct_change().add(1).cumprod()])
            exp = pd.Series(cumprods.index.map(lambda x:1/x.timetuple().tm_yday), index=cumprods.index, name='1/Days')
            returns = pd.concat([returns, cumprods.loc[start:end].pow(exp, axis='index')])

    return cumprods, returns

def returns_matrix_to_returns(periods, returns_matrix):
    '''
    periods: Benchmark().find_periods generated start/end datetime pairs that indicate rebalancing periods
    returns_matrix: Benchmark().make_returns_matrix() generated dataframe
    
    returns both each asset's respective returns and all assets' returns
    '''
    returns = pd.DataFrame()
    for start, end in tqdm(periods):
        returns = pd.concat([returns, returns_matrix.loc[end]], axis='columns')
    return returns.T.sum(), returns.T.sum(axis='columns')

def normality_tests(assets):
    ''' Tests for normality distribution of given data set.
    Parameters
    ==========
    assets: dataframe
    object to generate statistics on
    '''
    for asset in assets:
        '''asset should be input in the following fuctions as a ndarray'''
        print('\n', asset)
        print('Skew of data set %14.3f' % scs.skew(assets[asset].values))
        print('Skew test p-value %14.3f' % scs.skewtest(assets[asset].values)[1])
        print('Kurt of data set %14.3f' % scs.kurtosis(assets[asset].values))
        print('Kurt test p-value %14.3f' % scs.kurtosistest(assets[asset].values)[1])
        print('Norm test p-value %14.3f' % scs.normaltest(assets[asset].values)[1])

def graph_normality(assets, currencies):
    '''
    assets: dataframe for assets' daily prices
    currencies: cash or currency assets' tickers in a list
    graphically tests if a dataframe is normally distributed (Are returns of assets log normal?)
    '''
    cash, noncash = benchmark.columns_except(currencies, assets)

    filename = '_'.join(assets.columns).replace('/','')
    np.log(assets.pct_change())[noncash].replace([np.inf,-np.inf], np.nan).hist(bins=30)
    plt.savefig(f'log_normal_{filename}.png')
    
    for col in assets.columns:
        sm.qqplot(np.log(assets[col].pct_change()).replace([np.inf,-np.inf], np.nan).dropna(), line='s')
        plt.title(f'{col}')
        filename = col.replace('/','')
        plt.savefig(f'qqplot_{filename}.png')
        
def efficient_frontier(assets):
    days = len(assets)
    print('Finding annual periods of the investment')
    periods = benchmark.find_periods(assets)
    returns = []
    volatility = []
    print('Monte Carlo Simulation in progress')
    for _ in tqdm(range(2500)):            
        weights = np.random.random(len(assets.columns))
        weights /= np.sum(weights)
        
        weighted_returns = pd.DataFrame()
        for start, end in periods:
            weighted_returns = pd.concat([weighted_returns, (assets.pct_change()*weights).groupby(assets.index.year).get_group(end.year).sum()], axis='columns')
            
        variance = math.sqrt(np.dot(weights.T, np.dot(assets.cov()*days,weights)))
        returns.append(weighted_returns.sum().mean())
        volatility.append(variance)
    returns = np.array(returns)
    volatility = np.array(volatility)
    filename = '_'.join(assets.columns)
    plt.figure(figsize=(10,6))
    plt.scatter(volatility, returns, c=returns/volatility, marker='o', cmap='coolwarm')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.savefig(f'efficient_frontier_{filename}.png')
    return returns, volatility  

def get_volatility(weights, values_matrix):
    '''
    weights: weights of investment for each asset - pandas.Series
    values_matrix: holdings_matrix * assets' price matrix - pandas.DataFrame
    returns variance of assets' returns as volatility
    '''
    days = len(values_matrix)
    return math.sqrt(np.dot(weights.T, np.dot(values_matrix.cov()*days,weights)))

def get_sharpe(returns, volatility):
    '''
    returns: total returns of assets, the results of Benchmark.returns_matrix_to_returns(Benchmark.make_returns_matrix())
    returns Sharpe Ratio = returns/variance = returns/volatility
    '''
    return returns/volatility


periods = find_periods(assets)
holdings_matrix = make_holdings_matrix(weights, holdings, periods, assets)

values_matrix = holdings_matrix*assets
cumprods_matrix, returns_matrix = make_returns_matrix(periods, values_matrix, 'annual')
asset_returns, total_returns = returns_matrix_to_returns(periods, returns_matrix)

normality_tests(assets)
graph_normality(assets, [cash])

normality_tests(eikon)
graph_normality(eikon, [])

returns, volatility = efficient_frontier(eikon)
sharpe = get_sharpe(total_returns.sum(), get_volatility(weights, values_matrix))



days = len(assets)
print('Finding annual periods of the investment')
periods = benchmark.find_periods(assets)
returns = []
volatility = []
print('Monte Carlo Simulation in progress')
for _ in tqdm(range(2500)):            
    weights = np.random.random(len(assets.columns))
    weights /= np.sum(weights)
    
    weighted_returns = pd.DataFrame()
    for start, end in periods:
        weighted_returns = pd.concat([weighted_returns, (assets.pct_change()*weights).groupby(assets.index.year).get_group(end.year).sum()], axis='columns')
        
    variance = math.sqrt(np.dot(weights.T, np.dot(assets.cov()*days,weights)))
    returns.append(weighted_returns.sum().mean())
    volatility.append(variance)

    
sns.lineplot(total_returns.loc[datetime(1980,1,2):])
plt.savefig('total_returns.png')
sns.lineplot(cumprods_matrix.loc[datetime(1980,1,2):])
plt.savefig('cumprods_matrix.png')
sns.pairplot(assets)
plt.savefig('pairplot.png')
sns.barplot(cumprods_matrix.loc[datetime(1980,1,2):])
plt.savefig('barplot.png')
sns.heatmap(assets.corr(), annot=True)
plt.savefig('corr.png')
