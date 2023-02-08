import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os, csv
from tqdm import tqdm
sns.set_theme(style='ticks')

# with open('./25_Portfolios/25_value_weighted.csv') as csvfile:
#     file = csv.reader(csvfile)

# When the csv file is manually preedited, use the following commented-out statements
# df = pd.read_csv('./25_Portfolios/25_Portfolios_5x5.CSV')
# df.index = df.index.astype('str').map(lambda x:datetime.strptime(x,'%Y%m'))

# When the csv file is not preprocessed manually, use the following. The following are not finished.
df = pd.read_csv('./25_Portfolios/25_Portfolios_5x5.CSV', header=9, skiprows=lambda x: 'Average' in str(x), skipfooter=1)
breaks = df.index[df['SMALL LoBM']=='SMALL LoBM']
breaks = breaks.insert(0,0)
data = {}
for idx in range(len(breaks)-1):
    data[idx] = df[breaks[idx]:breaks[idx+1]]
for key, values in data.items():
    data[key] = values.reset_index()
data

# np.where(df.index=='NaT')
# df.loc['NaT']
# df.loc[:'NaT']

# Before completing preprocessing data, use the first dataframe to analyze for now
df = df[:1158]
df['Unnamed: 0'] = df['Unnamed: 0'].astype('str')
df.index = [datetime.strptime(x.strip(),'%Y%m') if len(x.strip())==6 else datetime.strptime(x.strip(),'%Y') if len(x.strip())==4 else np.NaN for x in df['Unnamed: 0'].values]
df = df.drop(['Unnamed: 0'], axis='columns')
df = df.astype('float')

assets = ['SMALL LoBM', 'SMALL HiBM', 'ME2 BM1', 'ME5 BM4', 'BIG LoBM', 'BIG HiBM']
cumprods = ['SL', 'SH', 'ML', 'MH', 'BL', 'BH']
returns = [i+'R' for i in cumprods]


# produce yearly stats    
annual = pd.DataFrame()
for year in tqdm(df.index.year):
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
sns.lineplot(annual[cumprods].mean(axis='columns'))

plt.show()


# alternative way to implement the above
annual_easy = pd.DataFrame()
for yearly in tqdm(df.index.year):
    annual_easy = pd.concat([annual_easy, df.groupby(df.index.year)[assets].get_group(yearly).divide(100).add(1).cumprod()])
sns.lineplot(annual_easy.mean(axis='columns'))
plt.show()

# The following does not generate the right values. Because
# 'df.groupby(df.index.year)[assets].apply(lambda x:x)'  is same as 'df' 
# annual_easy = df.groupby(df.index.year)[assets].apply(lambda x:x).divide(100).add(1).cumprod()
# sns.lineplot(annual_easy.mean(axis='columns'))
# plt.show()

total = df[assets]
total = total.divide(100).add(1).cumprod()
sns.lineplot(total.mean(axis='columns'))
plt.show()

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
