import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os, csv
sns.set_theme(style='ticks')

# with open('./25_Portfolios/25_value_weighted.csv') as csvfile:
#     file = csv.reader(csvfile)

df = pd.read_csv('./25_Portfolios/25_value_weighted.csv')
df.index = df.index.astype('str').map(lambda x:datetime.strptime(x,'%Y%m'))

assets = ['SMALL LoINV', 'SMALL HiINV', 'ME1 INV2', 'ME5 INV4', 'BIG LoINV', 'BIG HiINV']
cumprods = ['SL', 'SH', 'ML', 'MH', 'BL', 'BH']
returns = [i+'R' for i in cumprods]

# produce yearly stats    
annual = pd.DataFrame()
for year in df.index.year:
# in case you want to fancy index multiple years, i.e. every 3 year, use the following
# for years in df.index.year[::3]:
    yearly = df.loc[df.index.year==year]
    for cumprod, asset in zip(cumprods, assets):
        yearly[cumprod] = yearly[asset].divide(100).add(1).cumprod()
        # yearly[cumprod+'0'] = [yearly[cumprod].values[0] for _ in range(len(yearly))]
        # yearly[cumprod+'R'] = (yearly[cumprod]/yearly[cumprod+'0'])**(1/yearly.index.month) - 1
        yearly[cumprod+'R'] = (yearly[cumprod]/yearly[cumprod].values[0])**(1/yearly.index.month) - 1
    annual = pd.concat([annual, yearly[assets+cumprods+returns]])    

ret = annual[returns]

# gather all December stats for annual stats
decembers = ret.loc[ret.index.month==12]
decembers.sum()
decembers.sum(axis='columns')

sns.boxplot(decembers.sum(axis='columns'),palette='Blues')
plt.savefig('annual.png')
sns.boxplot(decembers.sum())
plt.savefig('annual_by_asset.png')

# produce entire period stats
for cumprod, asset in zip(cumprods, assets):
    df[cumprod] = df[asset].divide(100).add(1).cumprod()

sns.lineplot(df[cumprod])
plt.savefig('all.png')
sns.pairplot(df[cumprods])
plt.savefig('all_pairs.png')
sns.heatmap(df[cumprods].corr(), annot=True)
plt.savefig('all_corr.png')
