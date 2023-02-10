from ipyparallel import Client
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

start = datetime(1789, 5, 1)

df = {}
XAUUSDurl = 'https://stooq.com/q/d/l/?s=xauusd&i=d'
df['XAU/USD'] = pd.read_csv(XAUUSDurl, parse_dates=['Date'], index_col=0).sort_index(ascending=False)
df['S&P500'] = pdr.DataReader('^SPX', 'stooq', start, datetime.today())
df['US10Y'] = pdr.DataReader('10USY.B', 'stooq', start, datetime.today())

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

cumprods = assets.pct_change().add(1).cumprod()
cumprods = cumprods[1:]
sns.lineplot(cumprods.sum(axis='columns'))
plt.show()

# make price matrix and weight matrix
