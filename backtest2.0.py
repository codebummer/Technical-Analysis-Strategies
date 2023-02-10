from ipyparallel import Client
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math

start = datetime(1789, 5, 1)

df = {}
XAUUSDurl = 'https://stooq.com/q/d/l/?s=xauusd&i=d'
df['S&P500'] = pdr.DataReader('^SPX', 'stooq', start, datetime.today())
df['US10Y'] = pdr.DataReader('10USY.B', 'stooq', start, datetime.today())
df['XAU/USD'] = pd.read_csv(XAUUSDurl, parse_dates=['Date'], index_col=0).sort_index(ascending=False)

# make isoranged dataframes
cut = max([df[key].index[-1] for key in df.keys()])
for asset in df.keys():
    df[asset] = df[asset][:cut]

# make isodated dataframes
def make_isodate(dic, index):
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

union = set()
for asset in df.keys():
    df[asset].index.map(lambda x:union.add(x))

union = list(union)
union = pd.Series(union, name='Date').sort_values(ascending=False)
assets = make_isodate(df, union)
assets.isna().sum()
assets.fillna(method='bfill').isna().sum()
assets = assets.fillna(method='bfill')
assets.isna().sum()
assets.fillna(method='ffill').isna().sum()
assets = assets.fillna(method='ffill')
assets['US10Y'] = assets['US10Y'].divide(100).add(1)
assets['USD'] = [1 for _ in range(len(assets))]

cumprods = assets.pct_change().add(1).cumprod()
cumprods = cumprods[1:]
sns.set_theme(style='ticks')
sns.lineplot(cumprods.sum(axis='columns'), palette='rocket_r')
plt.show()

# make price matrix and weight matrix of the portfolio

# initial amount of investment
invested = 30_000
# weights for assets in ratio
weights = pd.Series({'S&P500':0.3, 'US10Y':0.5, 'XAU/USD':0.15, 'USD':0.05}, name='Weights')
# initialize holdings that represents stock numbers held, following the ratio
holdings = {}
for asset in assets.keys():
    if asset == 'S&P500':
        holdings[asset] = math.floor(invested*weights[asset]/(assets[asset].values[0]*0.1)) # VOO is approximately 9.1% of S&P500
    elif asset == 'US10Y':        
        holdings[asset] = math.floor(invested*weights[asset]/10) # Minimum amount to buy US10Y is USD 10
    elif asset == 'XAU/USD':
        holdings[asset] = math.floor(invested*weights[asset]/(assets[asset].values[0]*0.1)) # GLD is approximately 9.3% of XAU/USD
    else:
        holdings[asset] = math.floor(invested*weights[asset]/assets[asset].values[0])
holdings = pd.Series(holdings, name='Holdings')



assets.groupby(assets.index.year).get_group(2022)*holdings

holdings*assets.iloc[0,:].sum()
prices=holdings*assets.iloc[0,:]
prices/prices.sum()-weights

prices = assets.iloc[0,:]
values = holdings*prices
values['US10Y'] = holdings['US10Y']*10
values/values.sum()-weights

holdings
