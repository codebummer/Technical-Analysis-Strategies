import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3


with sqlite3.connect(r'D:\myprojects\TradingDB\삼성전자_주식틱차트1틱_2022_11_28.db') as file:
    # To read from sql by pandas you have to include the table name with square brackets
    df = pd.read_sql('SELECT * FROM [삼성전자 주식틱차트1틱]', file) 
# Rename the columns
df = df[['체결시간', '시가', '고가', '저가', '현재가', '거래량']]
df[df.columns] = df[df.columns].apply(lambda x:x.str.strip('-+'))
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df.index = df['Date']
df.drop(columns=['Date'], inplace=True)
df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S')
df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
df.index = pd.to_datetime(df.index)
# df[df.columns] = df[df.columns].apply(lambda x:x.astype('float64'))
df = df.astype('float64')


df_realtime = pd.DataFrame([])
strategies = []

currentcash = 0 
mfi_df = pd.DataFrame([])
BUY_REALTIME, SELL_REALTIME = 0, 0
orders_realtime = pd.DataFrame([])

def Bollinger(): # modified by me -> buy and sell conditions reversed and it resulted better
    global df, strategies

    strategies.append('Bollinger')    

    df['MA20'] = df.Close.rolling(window=20).mean()
    df['STD'] = df.Close.rolling(window=20).std()
    df['Upper'] = df.MA20 + 2 * df.STD
    df['Lower'] = df.MA20 - 2 * df.STD
    df['PB'] = (df.Close - df.Lower) / (df.Upper - df.Lower)
    df['Bandwidth'] = (df.Close - df.Lower) / df.MA20 * 100
    df['SQZ'] = ((df.Upper - df.Lower) / df.MA20) * 100
    
    buy, sell = [], []
    # for i in range(len(df.Close)):
    #     if df.PB.values[i] > 0.8: 
    #         buy.append(i)
    #     elif df.PB.values[i] < 0.2:
    #         sell.append(i)
 
    for i in range(len(df.Close)):
        if df.PB.values[i] < 0.2 and df.SQZ.values[i] < 10: 
            buy.append(i)
        elif df.PB.values[i] > 0.8:
            sell.append(i) 
            
    return buy, sell
    
def MFI(): # modified by me -> buy and sell conditions are reversed and it resulted better.
    global df, strategies

    strategies.append('MFI')
    
    df['TP'] = (df.High + df.Low + df.Close) / 3
    df['PMF'] = 0
    df['NMF'] = 0    
       
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
    # for i in range(len(df.Close)):
    #     if df.MFR10.values[i] > 80:
    #         buy.append(i)
    #     elif df.MFR10.values[i] < 20:
    #         sell.append(i)

    # for i in range(len(df.Close)-1):
    #     if df.MFR10.values[i] < 80 and df.Close.values[i] > df.Close.values[i+1]:
    #         buy.append(i)
    #     elif df.MFR10.values[i] > 20 and df.Close.values[i] < df.Close.values[i+1]:
    #         sell.append(i)

    for i in range(len(df.Close)-1):
        if df.MFR10.values[i] < 20:
            buy.append(i)
        elif df.MFR10.values[i] > 80:
            sell.append(i)

    return buy, sell
   
def RSI(): #modified by me -> changed the numbers by which buy and sell are decided to make the number of transactions bigger
    global df, strategies

    strategies.append('RSI')

    df['Diff'] = df.Close.diff(1)
    df['Gain'] = df.Diff.clip(lower=0).round(2)
    df['Loss'] = df.Diff.clip(upper=0).abs().round(2)
    df['AvgGain'] = df.Gain.rolling(window=10).mean()
    df['AvgLoss'] = df.Loss.rolling(window=10).mean()
    df['RSI'] = 100 - 100 / (1 + df.AvgGain / df.AvgLoss)

    buy, sell = [], []
    # for i in range(len(df.Close)):
    #     if df.RSI.values[i] < 10:
    #         buy.append(i)
    #     elif df.RSI.values[i] > 70:
    #         sell.append(i)
            
    for i in range(len(df.Close)):
        if df.RSI.values[i] < 20:
            buy.append(i)
        elif df.RSI.values[i] > 60:
            sell.append(i)
    
    return buy, sell

def Volume_RSI():
    global df, strategies

    strategies.append('Volume RSI')
    
    df['VolDiff'] = df.Volume.diff(1)
    df['VolGain'] = df.VolDiff.clip(lower=0)
    df['VolLoss'] = df.VolDiff.clip(upper=0).abs()
    df['VolAvgGain'] = df.VolGain.rolling(window=10).mean()
    df['VolAvgLoss'] = df.VolLoss.rolling(window=10).mean()
    df['VolRSI'] = 100 - 100 / (1 + df.VolAvgGain / df.VolAvgLoss)
    
    buy, sell = [], []
    for i in range(len(df.Close)):
        if df.VolRSI.values[i] < 20:
            buy.append(i)
        elif df.VolRSI.values[i] > 60:
            sell.append(i)
    
    return buy, sell

def MA_Line():
    global df, strategies

    strategies.append('Moving Average Lines')

    df['MA3'] = df.Close.rolling(window=3).mean()
    df['MA5'] = df.Close.rolling(window=5).mean()
    df['MA10'] = df.Close.rolling(window=10).mean()
    df['MA20'] = df.Close.rolling(window=20).mean()
    df['MA60'] = df.Close.rolling(window=60).mean()
    df['MA120'] = df.Close.rolling(window=120).mean()
    df['MA240'] = df.Close.rolling(window=240).mean()
    df['VolChangePer'] = df.Volume.pct_change(1)

    sell_counter = 0
    buy, sell = [], []
    for i in range(len(df.Close)):
        if df.MA60.values[i] > df.MA20.values[i] and df.MA20.values[i] > df.MA10.values[i] and df.MA10.values[i] > df.MA5.values[i]:
            buy.append(i)  
        # elif len(buy) > len(sell): 
        #     # The above statement shoulbe come before the statement in the next line
        #     # because buy[sell_counter] causes an index error when sell_counter is 0.

        #     if df.Close.values[i]/df.Close.values[buy[sell_counter]] - 1 >= 0.1:
        #         sell.append(i)
        #         sell_counter += 1
    

    # sell is filled with entire index numbers of records
    # This will make other strategies used together decide selling timing 
    sell = [i for i in range(len(df.Close))]

    return buy, sell    

def MA_Line_Volume(): # Selling timing will be decided by other strategies used at the same time
    global df, strategies

    strategies.append('Moving Average Lines with Volume')

    df['MA3'] = df.Close.rolling(window=3).mean()
    df['MA5'] = df.Close.rolling(window=5).mean()
    df['MA10'] = df.Close.rolling(window=10).mean()
    df['MA20'] = df.Close.rolling(window=20).mean()
    df['MA60'] = df.Close.rolling(window=60).mean()
    df['MA120'] = df.Close.rolling(window=120).mean()
    df['MA240'] = df.Close.rolling(window=240).mean()
    df['VolChangePer'] = df.Volume.pct_change(1)

    sell_counter = 0
    buy, sell = [], []
    for i in range(len(df.Close)):
        if df.MA60.values[i] > df.MA20.values[i] and df.MA20.values[i] > df.MA10.values[i] and df.MA10.values[i] > df.MA5.values[i]:
            if df.Open.values[i] >= df.MA3.values[i] and df.VolChangePer.values[i] >= 0.4:
                buy.append(i)  
        elif len(buy) > len(sell):
            if df.Close.values[i]/df.Close.values[buy[sell_counter]] - 1 >= 0.1:
                sell.append(i)
                sell_counter += 1
    
    return buy, sell


def MACD():
    global df, strategies

    strategies.append('MACD')

    fast = 12
    slow = 26
    signal = 9
    # weighting_decrese = 2 / (fast + 1)

    df['EMA12'] = df.Close.ewm(span=fast, adjust=False).mean()
    df['EMA26'] = df.Close.ewm(span=slow, adjust=False).mean()
    df['MACD'] = df.EMA12 - df.EMA26
    df['Signal'] = df.MACD.ewm(span=signal, adjust=False).mean() #This is same as EMA9 of MACD
    df['Histogram'] = df.MACD - df.Signal

    buy, sell = [], []
    for i in range(len(df.Close)):
        if df.MACD.values[i] > df.Signal.values[i]:
            buy.append(i)
        elif df.MACD.values[i] < df.Signal.values[i]:
            sell.append(i)
    
    return buy, sell


def common_orderID(*orderIDs_for_all): #returns common buy and sell indices
    buy = [orderIDs_per_strategy[0] for orderIDs_per_strategy in orderIDs_for_all[0]]
    sell = [orderIDs_per_strategy[1] for orderIDs_per_strategy in orderIDs_for_all[0]]
    
    def find_common(all):
        seen = set()
        common = set()
        for x in all:
            if x in seen:
                common.add(x)
            else:
                seen.add(x)                
        return list(common)
        
    return find_common([i for sub in buy for i in sub]), find_common([i for sub in sell for i in sub])

def make_order(orderID, shares): 
    #simulates buy and sell orders in a sequential time line
    #returns buy and sell index, considering no short order is possible
    global df, currentcash
    BUY, SELL = 0, 0
    buy, sell = [], []
    orders_made = [[],[]]
    dailyinvestment = 0
    accruedinvestment = 0
    
    if len(orderID) == 1:
        orderID = orderID[0]       

    for i in range(len(df.Close)):
        #orderID[0]: a list of buy orders
        if i in orderID[0] and currentcash > 0 and accruedinvestment > 10_000_000: 
            if dailyinvestment < 3_000_000:
                continue
            else:
                lastbuy = df.index.values[i]
                BUY += 1
                buy.append(df.Close.values[i])
                orders_made[0].append(i)
                dailyinvestment += df.Close[i] * shares
                accruedinvestment += df.Close[i] * shares
                # print(f'BUY  #{BUY} at {df.Close[i]}KRW')
        elif i in orderID[1] and BUY > SELL: # orderID[1]: a list of sell orders
            SELL += 1
            sell.append(df.Close.values[i])
            orders_made[1].append(i)
            if 
            dailyinvestment -= df.Close[i] * shares
            # print(f'SELL #{SELL} at {df.Close[i]}KRW')
        
            
    return orders_made, buy, sell        

 
def calc_returns(orders_made):
    global df, eval

    orders_made_ID, buy, sell = orders_made[0], orders_made[1], orders_made[2]    
    not_sold, buy = orders_made_ID[0][len(sell):], buy[:len(sell)]
    print(f'Unsold orders are BUY #{not_sold}')
    
    eval = pd.DataFrame({
        'Buy' : buy,
        'Sell' : sell
    })
    eval['Returns'] = (eval.Sell / eval.Buy - 1) * 100
    
def calc_total_returns(shares):
    global eval, strategies
        
    total_buy = eval.Buy.sum() * shares
    total_sell = eval.Sell.sum() * shares
    print(f'Strategies Implemented: {strategies}')
    print(f'Total Money Transacted: {total_buy:,}KRW, Total Returns: {total_sell-total_buy:,}KRW or {(total_sell/total_buy-1)*100:,.0000}%')


def visualize(orders_made):
    global df, strategies

    buy_index = orders_made[0][0]
    sell_index = orders_made[0][1]

    fig = plt.figure(figsize=(12,8))
    top_axes = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=4)
    plt.title(label=strategies, loc='right')   
    bottom_axes = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
    bottom_axes.get_yaxis().get_major_formatter().set_scientific(False)    
    top_axes.plot(df.index, df.Close, label = 'Close')
    top_axes.plot(df.index[buy_index], df.Close[buy_index], '^r')
    top_axes.plot(df.index[sell_index], df.Close[sell_index], 'vb')
    bottom_axes.plot(df.index, df.Volume) 
    plt.show()
    
    strategies = []


def order_easy(strategies, money, shares):
    global currentcash
    currentcash = money
    if len(strategies) > 1:
        common_orders_ID = common_orderID(strategies)
        orders_made = make_order(common_orders_ID, shares)

    else:
        orders_made = make_order(strategies, shares)

    calc_returns(orders_made)
    calc_total_returns(shares)
    visualize(orders_made)



# strategies should be input in the form of a list for order_easy
# or, order_easy will think the second strategy as the share nunmer

# order_easy([Bollinger(), MFI()], 1_000_000, 10)
# order_easy([Bollinger()], 1_000_000, 10)
# order_easy([MFI()], 1_000_000, 10)
# order_easy([RSI()], 1_000_000, 10)
# order_easy([Bollinger(), MFI(), RSI()], 1_000_000, 10)
# order_easy([MFI(), RSI()], 1_000_000, 10)
# order_easy([Bollinger(), RSI()], 1_000_000, 10)
# order_easy([MACD()], 1_000_000, 10)
# order_easy([Bollinger(), MFI(), RSI(), MACD()], 1_000_000, 10)
# order_easy([Bollinger(), MACD()], 1_000_000, 10)
# order_easy([MFI(), MACD()], 1_000_000, 10)
# order_easy([RSI(), MACD()], 1_000_000, 10)
# order_easy([Bollinger(), RSI(), MACD()], 1_000_000, 10)
# order_easy([Volume_RSI()], 1_000_000, 10)
# order_easy([RSI(), Volume_RSI()], 1_000_000, 10)
# order_easy([MFI()], 1_000_000, 10)
# order_easy([MA_Line(), MFI()], 1_000_000, 10)
# order_easy([MFI(), MA_Line_Volume()], 1_000_000, 10)
# order_easy([MA_Line(), Bollinger()], 1_000_000, 10)
# order_easy([MACD(), MA_Line_Volume()], 1_000_000, 10)
# order_easy([MACD(), MA_Line()], 1_000_000, 10)
# order_easy([MA_Line(), MFI(), Bollinger()], 1_000_000, 10)
# order_easy([MA_Line_Volume()], 1_000_000, 10)
# order_easy([Bollinger(), MA_Line()], 1_000_000, 10)
# order_easy([Bollinger(), MA_Line(), MFI()], 1_000_000, 10)
order_easy([Bollinger(), MA_Line(), MFI(), RSI()], 1_000_000, 10)

# pear_corr = df.corr(method='pearson')
# linedup = pear_corr.stack().sort_values(ascending=False)
# linedup = linedup[[linedup[i] != 1 for i in range(len(linedup))]]
# linedup[[0, -1]]

# pear_corr.style.background_gradient(cmap='Green', axis=0)

# pd.plotting.scatter_matrix(df, figsize=(20,20))
# plt.show()
