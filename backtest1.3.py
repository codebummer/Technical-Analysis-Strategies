import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas_datareader.data as pdr
import os, csv
from tqdm import tqdm
import sqlite3
import xmltodict, json
sns.set_theme(style='ticks')

# KOSPI listed stocks' data
with sqlite3.connect('./MarketDB/KoreaFins/fins_2022_Q1.db') as db:
    query = '''SELECT * FROM sqlite_master WHERE type='table';'''
    tables = db.cursor().execute(query).fetchall()

# collect all listed company names that have financial data
listed_corps = [table[1].split('_')[0] for table in tables]

with open(r'D:\myprojects\MarketDB\corpCode\CORPCODE.xml', 'r', encoding='utf-8') as file:
    codes = xmltodict.parse(file.read())

ticker_stock = {}
for stock in tqdm(codes['result']['list']):
    ticker_stock[stock['stock_code']] = stock['corp_name']
for stock in tqdm(codes['result']['list']):
    ticker_stock[stock['corp_name']] = stock['stock_code']
for key in list(ticker_stock.keys()):
    if ticker_stock[key] == None:
        ticker_stock.pop(key)
ticker_stock.pop(None)
with open(r'D:\myprojects\MarketDB\ticker_stock.json', 'w', encoding='utf-8') as file:
    # 'ensure_ascii=False' is required to keep Korean unbroken
    json.dump(ticker_stock, file, ensure_ascii=False)

data = {}
csvfiles = os.listdir(r'D:\myprojects\MarketDB\indices')
for csvfile in csvfiles:
    name = csvfile.strip('.csv').replace('_','/')
    data[name] = pd.read_csv(r'D:\myprojects\MarketDB\indices\\'+csvfile, index_col=0)
    data[name].index = data[name].index.map(lambda x:datetime.strptime(x,'%Y-%m-%d'))

strategy = ['KOSPI', 'S&P500', 'US2Y', 'USD/KRW', 'XAU/USD']
for asset in strategy:
    data[asset].sort_index(ascending=False, inplace=True)
data
[(n,len(data[n])) for n in strategy]
data['XAU/USD']
data['KOSPI']
data['S&P500']
data['US2Y']
data['US10Y']


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

# weighted annual returns
weights = [0.2, 0.25, 0.05, 0.05, 0.25, 0.05, 0.15]
sum(weights)

weight_matrix = {}
for idx, key in enumerate(annual_easy.columns):
    weight_matrix[key] = np.array([weights for _ in range(len(annual_easy))]).T[idx]
weight_matrix = pd.DataFrame(weight_matrix, index=annual_easy.index)
weighted = annual_easy*weight_matrix
sns.lineplot(weighted.sum(axis='columns'))
plt.show()

# total returns and total weighted returns
total = df[assets]
total = total.divide(100).add(1).cumprod()
total_weighted = total*weight_matrix
total_weighted.sum(axis='columns')
sns.lineplot(total.mean(axis='columns'))
sns.lineplot(total_weighted.sum(axis='columns'))
plt.show()

# gather all December stats for annual stats
decembers = ret.loc[ret.index.month==12]
decembers.sum()
decembers.sum(axis='columns')

# alternative way to implement the above
ret.groupby(ret.index.month).get_group(12)

# plotting
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
