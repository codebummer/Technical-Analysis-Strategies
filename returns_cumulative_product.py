import pandas as pd
import pandas_datareader.data as web
import numpy as np
from datetime import datetime 

start = datetime(1980, 1, 1)
end = datetime.today()

ticker = '005930'
df = web.DataReader(ticker, 'naver', start, end)

df = df.astype('float')
df = df.filter(['Close'])
df['Change'] = df['Close'].pct_change()
geomeans = df.Close[-1]/df.Close[0] - 1
goemeans

# change = df.pct_change()
# change.columns = ['Change']
# df = df.join(change)

df['CumProd'] = df.Change.add(1).cumprod()
cumProd = df.CumProd[-1] - 1

print(f'{geomeans - cumProd:.20f}')
