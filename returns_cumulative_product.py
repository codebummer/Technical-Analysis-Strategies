import pandas as pd
import pandas_datareader.data as web
import numpy as np
from datetime import datetime 

start = datetime(1980, 1, 1)
end = datetime.today()
years = end.year - start.year

ticker = '005930'
df = web.DataReader(ticker, 'naver', start, end)

df = df.astype('float')
df = df.filter(['Close'])
df['Change'] = df['Close'].pct_change()
geomeans = df.Close[-1]/df.Close[0] - 1

# change = df.pct_change()
# change.columns = ['Change']
# df = df.join(change)

df['CumProd'] = df.Change.add(1).cumprod()
df['Returns'] = df.CumProd.subtract(1)
cumprod = df.Returns[-1]

# annual_returns = np.roots([-years, df.CumProd[-1]]) - 1 #shoul be same as below, but somehow incorrect
annual_returns = np.power(df.CumProd[-1], 1/years) -1 #This is correct. This is called CAGR.

print(f'{geomeans - cumprod:.20f}')
