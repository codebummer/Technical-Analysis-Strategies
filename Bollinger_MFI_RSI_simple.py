import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

start = datetime(2020, 1, 1)
end = datetime(2022, 11, 11)
df = web.DataReader('005930.KS', 'yahoo', start, end)
df_realtime = pd.DataFrame([])

# bollinger_df = pd.DataFrame([])
mfi_df = pd.DataFrame([])
BUY_REALTIME, SELL_REALTIME = 0, 0
orders_realtime = pd.DataFrame([])

def Bollinger():
    global df
    
    df['MA20'] = df.Close.rolling(window=20).mean()
    df['STD'] = df.Close.rolling(window=20).std()
    df['Upper'] = df.MA20 + 2 * df.STD
    df['Lower'] = df.MA20 - 2 * df.STD
    df['PB'] = (df.Close - df.Lower) / (df.Upper - df.Lower)
    df['Bandwidth'] = (df.Close - df.Lower) / df.MA20 * 100
    df['SQZ'] = ((df.Upper - df.Lower) / df.MA20) * 100
    
    buy, sell = [], []
    for i in range(len(df.Close)):
        if df.PB.values[i] > 0.8: 
            buy.append(i)
        elif df.PB.values[i] < 0.2:
            sell.append(i)
            
    return buy, sell
    
def Bollinger_realtime(): 
    global df_realtime, bollinger_df
    #df_realtime here is OHLC information for realtime transactions
    
    ma20 = df_realtime.Close.rolling(window=20).mean()
    std = df_realtime.Close.rolling(window=20).std()
    upper = ma20 + 2 * std
    lower = ma20 - 2 * std
    pb = (df_realtime.Close - lower) / (upper - lower)
    add_df = pd.DataFrame([[ma20, std, upper, lower, pb]], columns = ['ma20', 'std', 'upper', 'lower', 'pb'])
    bollinger_df.append(add_df)

    if ma20 == 'NaN':
        return None
    if pb > 0.8:
        return 'BUY'
    elif pb < 0.2:
        return 'SELL'

def MFI():
    global df
    
    df['TP'] = (df.High + df.Low + df.Close) / 3
    df['PMF'] = 0
    df['NMF'] = 0
    
    # df['TP2'] = df.TP.shift(1)
    # df[df.TP < df.TP2] = 
        
    for i in range(len(df.Close)-1):
        if df.TP.values[i] < df.TP.values[i+1]:
            df.PMF.values[i+1] = df.TP.values[i+1] * df.Volume.values[i+1]
            df.NMF.values[i+1] = 0
        else:
            df.NMF.values[i+1] = df.TP.values[i+1] * df.Volume.values[i+1]
            df.PMF.values[i+1] = 0
    df['MFR'] = df.PMF.rolling(window=10).sum() / df.NMF.rolling(window=10).sum()
    df['MFR10'] = 100 - 100 / (1 + df.MFR)
    
    buy, sell = [], []
    for i in range(len(df.Close)):
        if df.MFR10.values[i] > 80:
            buy.append(i)
        elif df.MFR10.values[i] < 20:
            sell.append(i)
    
    return buy, sell
   
def MFI_realtime():    
    global df_realtime
    
    tp = (df_realtime.High + df_realtime.Low + df_realtime.Close) / 3
    if mfi_df.tp[-1] < tp:
        pmf = tp * df_realtime.Volume[-1]
        nmf = 0
    else:
        nmf = tp * df_realtime.Volume[-1]
        pmf = 0
    mfr = mfi_df.pmf.rolling(windows=10).sum() / mfi_df.nmf.rolling(window=10).sum()
    mfr10 = 100 - 100 / (1 + mfr)
    add_df = pd.DataFrame([[tp, pmf, nmf, mfr, mfr10]], columns = ['tp', 'pmf', 'nmf', 'mfr', 'mfr10'])
    mfi_df.append(add_df)
    
    if mfr == 'NaN':
        return None    
    if mfr10 > 80:
        return 'BUY'
    elif mfr10 < 20:
        return 'SELL'

def RSI():
    global df

    df['Diff'] = df.Close.diff(1)
    df['Gain'] = df.Diff.clip(lower=0).round(2)
    df['Loss'] = df.Diff.clip(upper=0).abs().round(2)
    df['AvgGain'] = df.Gain.rolling(window=10).mean()
    df['AvgLoss'] = df.Loss.rolling(window=10).mean()
    df['RSI'] = 100 - 100 / (1 + df.AvgGain/df.AvgLoss)

    buy, sell = [], []
    for i in range(len(df.Close)):
        if df.RSI.values[i] < 10:
            buy.append(i)
        elif df.RSI.values[i] > 70:
            sell.append(i)

    return buy, sell


def common_orderID(*orderIDs_for_all): #returns common buy and sell indices
    buy = [orderIDs_per_strategy[0] for orderIDs_per_strategy in orderIDs_for_all][0]
    sell = [orderIDs_per_strategy[1] for orderIDs_per_strategy in orderIDs_for_all][0]

    def find_common(all):
        seen = []
        for x in all:
            if x not in seen:
                seen.append(x)
        return seen
    
    return find_common(buy), find_common(sell)

def common_orderID_realtime():
    if Bollinger_realtime() == MFI_realtime() == 'BUY':
        make_order_realtime('BUY')
    elif Bollinger_realtime == MFI_realtime() == 'SELL':
        make_order_realtime('SELL')

def make_order(orderID): 
    #simulate buy and sell orders in a sequential time line
    #returns buy and sell index, considering no short order is possible
    global df
    BUY, SELL = 0, 0
    buy, sell = [], []
       
    for i in range(len(df.Close)):
        if i in orderID[0]: #orderID[0]: a list of buy orders
            BUY += 1
            buy.append(df.Close.values[i])
            print(f'BUY  #{BUY} at {df.Close[i]}KRW')
        elif i in orderID[1] and BUY > SELL: # orderID[1]: a list of sell orders
            SELL += 1
            sell.append(df.Close.values[i])
            print(f'SELL #{SELL} at {df.Close[i]}KRW')
            
    return orderID, buy, sell        

def make_order_realtime(order):
    global BUY_REALTIME, SELL_REALTIME
    
    if order == 'BUY':
        BUY_REALTIME += 1
        buy = df_realtime.Close[-1]
        sell = 0
    elif order == 'SELL' and BUY_REALTIME> SELL_REALTIME:
        SELL_REALTIME += 1
        sell = df_realtime.Close[-1]
        buy = 0
    
    add_df = pd.DataFrame([[df_realtime.index, buy, sell]], columns = ['Date', 'Buy', 'Sell'])
    orders_realtime.append(add_df)
 
  
def calc_returns(orders):
    global df, eval

    orderID, buy, sell = orders[0], orders[1], orders[2]    
    not_sold, buy = orderID[0][len(sell):], buy[:len(sell)]
    print(f'Unsold orders are BUY #{not_sold}')
    
    eval = pd.DataFrame({
        'Buy' : buy,
        'Sell' : sell
    })
    eval['Returns'] = (eval.Sell / eval.Buy - 1) * 100
    
def calc_total_returns(shares):
    global eval
        
    total_buy = eval.Buy.sum() * shares
    total_sell = eval.Sell.sum() * shares
    print(f'Total Investment: {total_buy:,}KRW, Total Returns: {total_sell-total_buy:,}KRW or {(total_sell/total_buy-1)*100:,.00f}%')

def order_easy(strategies, shares):
    if len(strategies) > 1:
        common_orders_ID = common_orderID(strategies[0])
        orders_made = make_order(common_orders_ID)

    else:
        orders_made = make_order(strategies[0])

    
    calc_returns(orders_made)
    calc_total_returns(shares)

order_easy([Bollinger(), MFI()], 20)
order_easy([Bollinger()], 20)
order_easy([MFI()], 20)
order_easy([RSI()], 20)
order_easy([Bollinger(), MFI(), RSI()], 20)
order_easy([MFI(), RSI()], 20)
order_easy([Bollinger(), RSI()], 20)
