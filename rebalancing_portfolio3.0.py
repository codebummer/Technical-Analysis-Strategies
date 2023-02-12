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


holdings_matrix = make_holdings_matrix(weights, holdings, periods, assets)