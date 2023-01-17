import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os, math
yf.pdr_override()
os.chdir(r'D:\myprojects')

class portfolio:
    def __init__(self, stocks):
        self.start = datetime(2006, 2, 6)
        self.end = datetime.today()
        self.invested = 10_000
        self.weights = stocks       

        self.years = tuple(range(self.start.year, self.end.year+1))  

        self.prices = self.get_prices(self.weights)
        self.days = self.find_days()

        self.weights = pd.Series(self.weights)
        self.FX = self.get_prices({'KRW=X':'USD/KRW'})

        self.execute()

    def get_prices(self, stocks):
        df = pd.DataFrame([])
        for ticker, stock in stocks.items():
            df[ticker] = (web.get_data_yahoo(ticker, self.start, self.end)['Close'])
        return df
    
    def find_days(self):
        days = []
        for year in self.years:
            days.append(len(self.prices.loc[self.prices.index.year==year]))
        return pd.Series(days, index=self.years)        

    def find_holdings(self):
        self.holdings = self.invested * self.weights/self.weights.sum() / self.prices.iloc[0,:]
        self.holdings = self.holdings.round().astype('int')
        diff = (self.weights/self.weights.sum() - self.holdings*self.prices.iloc[0,:]/(self.holdings*self.prices.iloc[0,:]).sum())*100
        if any(diff > 1):
            print('Warning: the following shows discrepancies in the designated asset allocation\n')
            print(diff.loc[diff>1])    
        self.holdings = self.holdings.to_frame(name=pd.Timestamp(self.start))
        self.holdings = self.holdings.T       

    def _find_yearly_prices(self):
        self.yearly_prices = pd.DataFrame(self.prices.iloc[0,:], columns=[self.prices.index[0]])
        for year in self.years:
            self.yearly_prices = pd.concat([self.yearly_prices, self.prices.loc[self.prices.index.year==year].iloc[-1,:]], axis='columns')
        self.yearly_prices = self.yearly_prices.T    
    
    def _rebalance_holdings(self):
        for yearend in self.yearly_prices.index[1:]:    
            add1 = pd.DataFrame(self.yearly_prices.loc[yearend] * self.holdings.iloc[-1,:], columns=[yearend]).T
            self.values = pd.concat([self.values, add1])    
            add2 = self.values.loc[yearend]/self.yearly_prices.loc[yearend]
            self.holdings = pd.concat([self.holdings, add2.round(decimals=1).astype('int').to_frame().T])
            
            off_values = self.values.loc[yearend] - self.weights/100*self.values.loc[yearend].sum()    
            off_qty = (off_values/self.yearly_prices.loc[yearend]).round(decimals=1).astype('int')     
            gains = off_qty.clip(lower=0) * self.yearly_prices.loc[yearend]
            losses = off_qty.clip(upper=0) * self.yearly_prices.loc[yearend]
            loss_unit_prices = off_qty.clip(upper=0).where(off_qty.clip(upper=0)==0,1) * self.yearly_prices.loc[yearend]
            loss_unit_prices = loss_unit_prices.loc[loss_unit_prices!=0].sort_values()
            
            # In case loos_unit_prices can be all zeroes
            if loss_unit_prices.shape[0]:
                min_asset, min_unit_price = loss_unit_prices.index[0], loss_unit_prices[0]   
            else:
                min_asset, min_unit_price = '', 0

            if gains.sum() < min_unit_price:
                continue  
            
            # In case off_qty.clip(upper=0).sum() is zero, add a very small numer to avoid dividing by zero.
            rebalance_ratio = off_qty.clip(upper=0) / (off_qty.clip(upper=0).sum()+0.0000000000000000000001)
            rebalance_qty = (gains.sum()*rebalance_ratio/self.yearly_prices.loc[yearend]).round(decimals=1).astype('int')
            rebalance_order = (rebalance_qty.abs()*self.yearly_prices.loc[yearend]).sort_values(ascending=False)
            rebalance_assets = rebalance_order.cumsum() < gains.sum() 
            # rebalance_qty.loc[rebalance_asset] can remove elements in case of an all zero filled series
            # instead, rebalance_qty * rebalance_asset will not remove elements
            self.holdings.loc[yearend] = self.holdings.loc[yearend] + rebalance_qty * rebalance_assets - off_qty.clip(lower=0)
    
    def _check_rebalancing(self):
        port = self.holdings * self.yearly_prices
        allocation_diff = (port.divide(port.sum(axis='columns'), axis='index') - self.weights/self.weights.sum()) * 100
        # allocation_diff.loc[allocation_diff.values>1][allocation_diff>1]     
        diff = allocation_diff.loc[allocation_diff.values>1]   
        if any(diff):
            print('Warning: the following asset allocations have more than 1 percent difference from original plans\n', diff)

        holdings_ratio = pd.DataFrame()
        for year in self.years:
            holdings_ratio = pd.concat([holdings_ratio,self.holdings.loc[self.holdings.index.year==year]/self.holdings.loc[self.holdings.index.year==year].sum(axis='columns')[0]])
        holdings_sum = holdings_ratio.sum(axis='columns')
        sum_discrepancies = holdings_sum.loc[holdings_sum!=1]
        if any(sum_discrepancies):
            print('\nWarning: the following asset allocations do not sum up to 100%\n', sum_discrepancies)

    def find_initial_values(self):
        self.values = self.holdings * self.prices.iloc[0,:]

    def _find_daily_values(self):
        self.daily_prices_FX_included = pd.DataFrame()
        for year in self.years:
            self.daily_prices_FX_included = pd.concat([self.daily_prices_FX_included, (self.prices.loc[self.prices.index.year==year]).multiply(self.FX.loc[self.FX.index.year==year], axis='index')])
        self.daily_prices_FX_included = self.daily_prices_FX_included.dropna()

        # daily prices * daily holdings (quantities of assets held)
        self.daily_portfolio_values = pd.DataFrame()
        self.daily_portfolio_values_FX_included = pd.DataFrame()
        for year in self.years:
            self.daily_portfolio_values = pd.concat([self.daily_portfolio_values, self.prices.loc[self.prices.index.year==year]*self.holdings.loc[self.holdings.index.year==year].iloc[0,:]])
            self.daily_portfolio_values_FX_included = pd.concat([self.daily_portfolio_values_FX_included, self.daily_prices_FX_included.loc[self.daily_prices_FX_included.index.year==year]*self.holdings.loc[self.holdings.index.year==year].iloc[0,:]])

    def visualize(self):
        weighted_prices = self.weights * self.prices
        daily_amounts = weighted_prices.sum(axis='columns')
        ratio = ((self.invested/len(self.prices.columns))/self.prices.iloc[0,:]).round()

        assets = {}
        for asset in self.prices.columns:
            assets[asset] = math.floor(self.invested/self.prices[asset][0])

        sns.set()
        fig, ax = plt.subplots(1)
        ax.plot((ratio*self.prices).sum(axis='columns'))
        ax.plot(self.daily_portfolio_values.sum(axis='columns'))
        for asset in assets.keys():
            ax.plot(self.prices[asset] * assets[asset])
        ax.legend(['Non-Weighted', 'Weighted'] + list(assets.keys()))
        plt.show()

    def execute(self):
        self.find_holdings()
        self.find_initial_values()
        self._find_yearly_prices()
        self._rebalance_holdings()
        self._check_rebalancing()
        self._find_daily_values()
        self.visualize()


portfolios = [
    {'SPY':60, 'IEF':40},
    {'SPY':70, 'IEF':30},
    {'SPY':80, 'IEF':20},
    {'SPY':90, 'IEF':10}, 
    {'SPY':60, 'TLT':40},
    {'SPY':70, 'TLT':30},
    {'SPY':80, 'TLT':20},
    {'SPY':90, 'TLT':10}
]

for port in portfolios:
    portfolio(port)
