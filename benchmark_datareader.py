import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
        download S&P500, US10Y, XAU/USD from stooq.com and
        returns a diction whose keys are asset names and whose values are their daily price or yield data'''
        dic = {}
        XAUUSDurl = 'https://stooq.com/q/d/l/?s=xauusd&i=d'
        dic['S&P500'] = pdr.DataReader('^SPX', 'stooq', self.start, datetime.today())
        dic['US10Y'] = pdr.DataReader('10USY.B', 'stooq', self.start, datetime.today())
        dic['XAU/USD'] = pd.read_csv(XAUUSDurl, parse_dates=['Date'], index_col=0).sort_index(ascending=False)
        return dic

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

    def yields_to_prices(self, invested, weight, yields, cumul=True):
        '''
        invested: total investment amount for all assets
        weight: weight for bonds investment to total investment in the working investment strategy
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
                add = _yields_to_prices(invested*weight, yields.groupby(yields.index.year).get_group(year))
                bond_prices = pd.concat([bond_prices, add])
        else:
            # bond prices when annually reinvested with past years' yearend prices 
            bond_prices = pd.Series()
            bond_reinvest = invested * weight
            for year in tqdm(list(set(yields.index.year))):
                add = _yields_to_prices(bond_reinvest, yields.groupby(yields.index.year).get_group(year))
                bond_prices = pd.concat([bond_prices, add])
                bond_reinvest = bond_prices[-1]
        return bond_prices
    
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
