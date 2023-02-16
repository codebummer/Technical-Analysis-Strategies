import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math, re
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

class Benchmark():
    def __init__(self):
        self.start = datetime(1789, 5, 1)
        self.dic = self.downloads()
        self.dic = self.make_isoranged(self.dic)        
        self.assets = self.make_isodated_dataframe(self.dic, self.get_isoindex(self.dic))
        self.assets = self.clean_up(self.assets)
    
    def get(self):
        '''returns a cleaned-up dataframe which contains S&P500, US10Y, XAU/USD daily prices and yields from stooq.com'''
        return self.assets

    def downloads(self):
        '''
        downloads S&P500, US10Y, XAU/USD from stooq.com and
        returns a diction whose keys are asset names and whose values are their daily price or yield data'''
        dic = {}
        XAUUSDurl = 'https://stooq.com/q/d/l/?s=xauusd&i=d'
        dic['S&P500'] = pdr.DataReader('^SPX', 'stooq', self.start, datetime.today())
        dic['US10Y'] = pdr.DataReader('10USY.B', 'stooq', self.start, datetime.today())
        dic['XAU/USD'] = pd.read_csv(XAUUSDurl, parse_dates=['Date'], index_col=0).sort_index(ascending=False)
        return dic

    def load_eikon(self):
        '''
        downloads AAPL, MSFT, INTC, AMZN, GS.N, SPY, .SPX, .VIX, EUR=, XAU=, GDX, GLD from yhilpisch's github account
        The data is used in the book, Python for Finance by Yves Hilpisch
        '''
        url = 'https://raw.githubusercontent.com/yhilpisch/py4fi/master/jupyter36/source/tr_eikon_eod_data.csv'      
        data = pd.read_csv(url, index_col=0, parse_dates=['Date'])  
        data.columns = ['AAPL', 'MSFT', 'INTC', 'AMZN', 'GS', 'SPY', 'SPX', 'VIX', 'EUR', 'XAU', 'GDX', 'GLD']
        return data
    
    def load_french(self, period='monthly'):   
        '''
        period: "daily" or "monthly" for daily or monthly prices dataframe
        returns two dataframes. However, the second "daily" dataframe is not cleaned up perfectly,
        to clean up, either remove the first two rows from it manually,
        or use Benchmark().load_french_daily() method
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

    def load_french_daily(self):    
        '''
        returns two daily dataframes.
        This method cannot be used generally. This is customized only for crawling thid dataset from this link.
        Use Benchmark().load_french() for more generalized crawling for other data from the same website.
        '''
        data = pd.read_csv('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_25_Portfolios_ME_BE-ME_daily_CSV.zip', index_col=0, header=11, parse_dates=True)
        '''returns two dataframes'''
        def _generate_index(index):
            index = index.map(lambda x:x.strip())
            dates = []
            breaks = []
            for idx in tqdm(range(len(index))):
                if index[idx] == '':
                    breaks.append(idx)
                    dates.append('')
                elif 'Average' in index[idx]:
                    dates.append('Average')
                else:               
                    dates.append(datetime.strptime(index[idx],'%Y%m%d'))
            return np.array(dates), breaks

        data.index, breaks = _generate_index(data.index)

        return data.iloc[:breaks[0]-1], data.iloc[breaks[0]+1:]

    # make isoranged dataframes
    def make_isoranged(self, dic):
        '''
        gets a dictionary input, makes their data in the same period, based on the shorted one, and returns the resulting dictionary
        dic: a diction whose keys are asset names and whose values are their daily price or yield data
        '''
        cut = max([dic[key].index[-1] for key in dic.keys()])
        for asset in dic.keys():
            dic[asset] = dic[asset][:cut]
        return dic
    
    # make an isoindex list
    def get_isoindex(self, dic):
        '''dic: a dictionary whose keys are asset names and whose values are their daily price or yield data'''
        union = set()
        for asset in dic.keys():
            dic[asset].index.map(lambda x:union.add(x))
        union = list(union)
        return pd.Series(union, name='Date').sort_values(ascending=False)

    # make isodated dataframes
    def make_isodated_dataframe(self, dic, index):
        '''
        dic: a dictionary whose keys are asset names and whose values are their daily price or yield data
        index: a list, numpy list, or pandas.Series that represents the date index that dic values should be aligned with
        returns one dataframe all combined, instead of a dictionary
        '''
        isodate = pd.DataFrame()
        for key, value in tqdm(dic.items()):
            add = {key:[]}
            for date in tqdm(index):
                try:
                    add[key].append(value.loc[date][0])
                except:
                    add[key].append(None)
            isodate = pd.concat([isodate,pd.DataFrame(add, index=index)], axis='columns')
        return isodate
    
    # clean up the data
    def clean_up(self, df):
        df.isna().sum()
        df.fillna(method='bfill').isna().sum()
        df = df.fillna(method='bfill')
        df.isna().sum()
        df.fillna(method='ffill').isna().sum()
        df = df.fillna(method='ffill')
        df['USD'] = [1 for _ in range(len(df))]
        df.sort_index(inplace=True)
        return df        

    def yields_to_prices(self, unitprice, yields, cumul=True):
        '''
        unitprice: principal price or face value for one bond
        yields: a pandas series or dataframe that holds daily bond yields with datetime.timestamp as index
        cumul: if True, calculate bond prices, reinvesting in bonds, every period
        '''
        def _yields_to_prices(assigned_amount, yields):
            # buying price = invest amount * (1 + yield rate)
            fixed_income = assigned_amount * (yields.values[0]/100+1)
            # calculate sequential prices = buy price * (1 + change rates in yields * -1)
            bond_prices = yields.pct_change().multiply(-1).add(1) * fixed_income        
            # fill the first row which got empty after pct_change() with the initial fixed income amount at the time of purchase
            bond_prices[0] = fixed_income
            return bond_prices

        if cumul == False:
            # make sure you don't loop through the entire list of (assets.index.year)
            # just loop through list(set(assets.index.year))
            bond_prices = pd.Series()
            for year in tqdm(list(set(yields.index.year))):
                add = _yields_to_prices(unitprice, yields.groupby(yields.index.year).get_group(year))
                bond_prices = pd.concat([bond_prices, add])
        else:
            # bond prices when annually reinvested with past years' yearend prices 
            bond_prices = pd.Series()
            bond_reinvest = unitprice
            for year in tqdm(list(set(yields.index.year))):
                add = _yields_to_prices(bond_reinvest, yields.groupby(yields.index.year).get_group(year))
                bond_prices = pd.concat([bond_prices, add])
                bond_reinvest = bond_prices[-1]
        return bond_prices
    
    
    def find_periods(self, df):
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

    def columns_except(self, takeouts, assets, isdf=True):
        '''
        takeout: a list of column name(s) you want to remove from assets' daily price dataframe
        assets: a Dataframe or a Seriesof assets' daily prices
        isdf: if True, assets is a dataframe and exeuctes dataframe.columns, 
        if False, assets is a Series, executes Series.index, not execute .columns to avoid an error
        returns (removing column labels, kept columns labels)
        to return indices together change the return value in the form of 
        (removing columns indices, removing column labels, kept columns indices, kept columns labels)
        '''
        if isdf == True:            
            keepers = list(assets.columns)
            if list(takeouts) == keepers:
                return takeouts, None
            elif list(takeouts) == [] or None:
                return None, keepers
            else:
                keepers_index = list(range(len(assets.columns)))
                takeouts_index = [keepers.index(takeout) for takeout in takeouts]
                for takeout_index, takeout in zip(takeouts_index,takeouts):
                    keepers_index.remove(takeout_index)
                    keepers.remove(takeout)        
        else:
            keepers = list(assets.index)
            if list(takeouts) == keepers:
                return takeouts, None
            elif list(takeouts) == [] or None:
                return None, keepers
            else:            
                keepers = list(assets.index)
                keepers_index = list(range(len(assets.index)))
                takeouts_index = [keepers.index(takeout) for takeout in takeouts]
                for takeout_index, takeout in zip(takeouts_index,takeouts):
                    keepers_index.remove(takeout_index)
                    keepers.remove(takeout)             
        # return takeouts_index, takeouts, keepers_index, keepers
        return takeouts, keepers

    def initial_holdings(self, invested, weights, assets, cash):
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

    # make weights matrix
    def make_weight_matrix(self, period, holdings, assets):
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
    def make_holdings_matrix(self, weights, holdings, periods, assets):
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
            
            holdings_matrix = pd.concat([holdings_matrix, self.make_weight_matrix((start,end), holdings, assets)])    
        
        return holdings_matrix

    def make_returns_matrix(self, periods, values_matrix, returns_periods='annualcum'):
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

    def returns_matrix_to_returns(self, periods, returns_matrix):
        '''
        periods: Benchmark().find_periods generated start/end datetime pairs that indicate rebalancing periods
        returns_matrix: Benchmark().make_returns_matrix() generated dataframe
        
        returns both each asset's respective returns and all assets' returns
        '''
        returns = pd.DataFrame()
        for start, end in tqdm(periods):
            returns = pd.concat([returns, returns_matrix.loc[end]], axis='columns')
        return returns.T.sum(), returns.T.sum(axis='columns')

    def normality_tests(self, assets):
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

    def graph_normality(self, assets, currencies):
        '''
        assets: dataframe for assets' daily prices
        currencies: cash or currency assets' tickers in a list
        graphically tests if a dataframe is normally distributed (Are returns of assets log normal?)
        '''
        cash, noncash = self.columns_except(currencies, assets)

        filename = '_'.join(assets.columns).replace('/','')
        np.log(assets.pct_change())[noncash].replace([np.inf,-np.inf], np.nan).hist(bins=30)
        plt.savefig(f'log_normal_{filename}.png')
        
        for col in assets.columns:
            sm.qqplot(np.log(assets[col].pct_change()).replace([np.inf,-np.inf], np.nan).dropna(), line='s')
            plt.title(f'{col}')
            filename = col.replace('/','')
            plt.savefig(f'qqplot_{filename}.png')

    def get_volatility(self, weights, values_matrix):
        '''
        weights: weights of investment for each asset - pandas.Series
        values_matrix: holdings_matrix * assets' price matrix - pandas.DataFrame
        returns variance of assets' returns as volatility
        '''
        days = len(values_matrix)
        return math.sqrt(np.dot(weights.T, np.dot(values_matrix.cov()*days,weights)))

    def get_sharpe(self, returns, volatility):
        '''
        returns: total returns of assets, the results of Benchmark.returns_matrix_to_returns(Benchmark.make_returns_matrix())
        returns Sharpe Ratio = returns/variance = returns/volatility
        '''
        return returns/volatility

    def efficient_frontier(self, assets):
        days = len(assets)
        print('Finding annual periods of the investment')
        periods = self.find_periods(assets)
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
            returns.append(weighted_returns.sum(axis='columns').mean())
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
    
    def plot_returns(self, assets, dates, cumul=True):
        '''
        assets: a dataframe that contains assets data to plot
        dates: a list of datetime values that point to start and end dates. Both values should be idential in assets.index values. use [datetime(start date), datetime(end date)]
        cumul: If True, plots cumulative product of changes in prices. If False, plots just changes in prices.
        
        to use this method easily, refer to the following example:
        assets.groupby(assets.index.year).get_group(2020) : list the date index to find the exact dates            
        plot_returns(assets[['S&P500', 'US10Y']], [datetime(2020,1,2),datetime(2020,12,31)], cumul=False)
        
        '''
        if cumul == True:
            for asset in assets.columns:
                if asset == 'US10Y':
                    sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change().multiply(-1).add(1).cumprod())
                else:
                    sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change().add(1).cumprod())
        else:
            for asset in assets.columns:
                if asset == 'US10Y':
                    sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change().multiply(-1))
                else:
                    sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change())
        plt.legend(labels=assets)
        plt.show() 
