import pandas as pd
import pandas_datareader.data as web
from pandas_datareader.naver import NaverDailyReader
import yfinance as yf
import numpy as np
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
yf.pdr_override()
os.chdir(r'D:\myprojects')

start = datetime(1987, 1, 1)
end = datetime.today()
years = tuple(range(start.year, end.year+1))

weights = {
    'KOSPI' : 70,
    '302190' : 30
}   

def fetch_prices(weights):
    df = pd.DataFrame()
    for ticker, stock in weights.items():
        # df[ticker] = web.get_data_yahoo(ticker, start, end)['Close']
        df[ticker] = NaverDailyReader(ticker, start, end).read().astype('float64')['Close']
    return df

prices = fetch_prices(weights)
weights = pd.Series(weights) 
invested = 10_000

def screen(prices):
    def _holloween(month):
        if month in [11,12,1,2,3,4]:
            return True
        else:
            return False
    screened = np.vectorize(_holloween)(prices.index.month)
    return prices.loc[screened]

def find_asset_ratio(invested, weights, prices):
    '''invested: current money amount, weights: planned money ratio, prices: current unit prices'''
    return (invested * weights / prices).round()

def find_off_ratio(past_ratio, holding_ratio, prices):
    '''past_ratio: past held asset units ratio, holding_ratio: currently held asset units ratio, prices: current unit prices'''
    (holding_ratio - past_ratio) * prices
    

def rebalance(prices):
    

holloween = screen(prices)
