import pandas as pd
import numpy as np
import requests
import json, os, sys
from datetime import datetime
sys.path.append(r'D:\myprojects\MarketDB')
from ECOS_key import ECOS_KEY
import seaborn as sns
import matplotlib.pyplot as plt

url = f'https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_KEY}/json/kr/1/100/403Y001/Q/1950Q1/2022Q4/*AA/?/?/?'
with requests.get(url) as response:
    exports =json.loads(response.content)

trade = {'Date':[], 'Export':[]}
converts = {'Q1':'01', 'Q2':'04', 'Q3':'07', 'Q4':'10'}
for quarter in exports['StatisticSearch']['row']:
    trade['Date'].append(datetime.strptime(quarter['TIME'][:-2]+converts[quarter['TIME'][-2:]], '%Y%m'))
    trade['Export'].append(float(quarter['DATA_VALUE']))

trade = pd.DataFrame(trade['Export'], index=trade['Date'], columns=['Export'])
sns.lineplot(trade)
plt.show()
