import pandas as pd
import numpy as np
from benchmark_datareader import Benchmark
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# initial amount of investment
invested = 30_000
# weights for assets in ratio
weights = pd.Series({'S&P500':0.3, 'US10Y':0.5, 'XAU/USD':0.15, 'USD':0.05}, name='Weights')

benchmark = Benchmark()
assets = benchmark.get()
# benchmark.plot_returns(assets,[datetime(2018,1,2),datetime(2021,12,31)])
bonds, nonbonds = benchmark.columns_except(['US10Y'], assets)
cash, noncash = benchmark.columns_except(['USD'], assets)
bonds = bonds[0]
cash = cash[0]
assets[bonds] = benchmark.yields_to_prices(10, assets[bonds], False)

# make price matrix and weight matrix of the portfolio
# initial amount of investment
invested = 30_000
# weights for assets in ratio
weights = pd.Series({'S&P500':0.3, 'US10Y':0.5, 'XAU/USD':0.15, 'USD':0.05}, name='Weights')

periods = []
for year in tqdm(list(set(assets.index.year))):
    dates = assets.groupby(assets.index.year).get_group(year).index
    periods.append((dates[0],dates[-1]))

count = 0
holdings = pd.Series({'S&P500':30, 'US10Y':50, 'XAU/USD':15, 'USD':5}, name='Holdings')
# benchmark.plot_returns(assets,[datetime(2000,1,3),datetime(2021,12,31)])
# assets.loc[datetime(2000,1,3):datetime(2021,12,31)]

# make weight matrix
def make_weight_matrix(period, holdings, assets):
    '''
    period: a tuple pair of start and end datetime.timestamp
    holdings: holdings ratio by which the weight matrix is generated
    assets: a dataframe which contains asset daily prices with date index from which the weight matrix will copy its date index
    returns a dataframe which is a weight matrix for the given period
    '''
    start, end = period
    dates = assets.loc[start:end].index
    return pd.DataFrame([holdings.values for _ in  dates], index=dates, columns=holdings.index)    

all_holdings = pd.DataFrame()

for start, end in tqdm(periods):
    # portfolio values at this period end = prices * numbers of stocks held from llast period end
    # holdings here refers to numbers of stocks from the last period end (last update)
    values = assets.loc[end] * holdings 
    
    # numbers of stocks based on thid period end market prices = portfolio values / current stock prices
    # holdings here refers to numbers of stocks based on the current market prices
    # In case all pandas.Series elements are in the form of 'xxxx.0', round() will cause an error, 
    # so cast it into integers without rounding    
    if any((values/assets.loc[end]).astype('str').str.isdecimal()) == False:
        holdings = (values/assets.loc[end]).astype('int64')
    else:
        holdings = (values/assets.loc[end]).round(decimals=1).astype('int64')

    # off-values = prices - sum of end prices * allocation ratio
    # prices lower or higher than allocation plans
    off_values = values - values.sum()*weights

    # off-quantities = off-values / end prices
    # asset quantities lower or higher than allocation plans, based on off-values
    off_qty = (off_values/assets.loc[end]).round(decimals=1).astype('int64')
    # positive/negative off-quantities
    gain_qty = off_qty.clip(lower=0)
    loss_qty = off_qty.clip(upper=0)

    # gains = (positive off-quantities & 0's for negative off-quantities) * end prices
    # money amounts of assets higher than allocation plans, based on off-quantities
    # keep negative off-quantities 0: use pandas.clip(lower=0), not dataframe.loc[year,off_qty<0], 
    # to keep the element numbers same, without only indexing positive elements
    gains = gain_qty * assets.loc[end]

    # losses = (negative off-quantities & 0's for positive off-quantities)* end prices
    # money amounts of assets higher than allocation plans, based on off quantities
    losses = loss_qty * assets.loc[end]

    # if minimum single unit price of negative off-quantity stock > gains, skip rebalancing
    # get single unit price of negative off-quantity stocks
    loss_unit_prices = loss_qty.where(loss_qty==0,1) * assets.loc[end]
    loss_unit_prices = loss_unit_prices.loc[loss_unit_prices!=0].sort_values()
    
    # if negative off-quantities are all 0's, keep the holding ratio, concatenate it, and move on to the next period
    if loss_qty.sum() == 0:
        all_holdings = pd.concat([all_holdings, make_weight_matrix((start,end), holdings, assets)])
        continue

        
    min_asset, min_unit_price = loss_unit_prices.index[0], loss_unit_prices[0]   
    
    # if sum of gains is less than minimal negative off-quantity price, 
    # keep the holding ratio, concatenate it, and move on to the next period  
    if gains.sum() < min_unit_price:
        all_holdings = pd.concat([all_holdings, make_weight_matrix((start,end), holdings, assets)])
        continue
        
    # rebalance ratio = negative off-quantity stocks / sum of negative off-quantity stocks
    rebalance_ratio = loss_qty / loss_qty.sum()   

    # rebalance quantities = quantities to restock = sum of gains * rebalance ratio / end prices
    check_decimals = gains.sum() * rebalance_ratio / assets.loc[end]
    if any(check_decimals.astype('str').str.isdecimal()) == False:
        rebalance_qty = check_decimals.astype('int64')
    else:
        rebalance_qty = check_decimals.round(decimals=1).astype('int64')    

    # actual prices to restock = rebalance quantities * end prices
    rebalance_order = (rebalance_qty.abs()*assets.loc[end]).sort_values(ascending=False)

    # threshold to buy restocking assets
    rebalance_assets = rebalance_order.cumsum() < gains.sum() 

    # correct holdings accoringly
    # restock_qty: additional stock quantities to buy vs. rebalance_qty: total stock quantities to meet and not to just buy
    restock_qty = rebalance_qty*rebalance_assets.where(rebalance_assets==True,0)
    holdings = holdings + restock_qty - gain_qty

    # extra cash left after buying and selling stocks to rebalance
    # extra cash amount = total gained amount - bought amount + sold amount
    # USD = gains - restock_qty*end_price + gain_qty*end_price = gains - (restock_qty-gain_qty)*end_price
    holdings[cash] += (gains-(restock_qty-gain_qty)*assets.loc[end]).sum()

    off_weights = assets.loc[end]*holdings/(assets.loc[end]*holdings).sum() - weights
    print('\noff_weights:\n', off_weights)
    
    noncash_off_values = off_weights.abs()[noncash]*(assets.loc[end,noncash].sum())
    noncash_off_assests = noncash_off_values < assets.loc[end,noncash]
    if any(noncash_off_assests):
        readjust, nonreadjust = benchmark.columns_except(noncash_off_assests.index, noncash_off_values, False)
        quotients = noncash_off_values[readjust] // assets.loc[end]
        holdings[cash] += quotients * assets.loc[end]       
     
    # if any(off_weights.abs()>0.1):
    if off_weights.abs()[cash]>0.1 and any(off_weights.abs()[noncash]*assets.loc[end,noncash].sum() > assets.loc[end,noncash]):
        count += 1
        print(f'Warning {count}th: rebalancing conducted has too large off-value(s)')
    
    # make a weight matrix and concatenate it 
    all_holdings = pd.concat([all_holdings, make_weight_matrix((start,end), holdings, assets)])

# returns = (all_holdings*assets).sum(axis='columns').pct_change().add(1).cumprod()
# sns.lineplot(returns)
# plt.ticklabel_format(style='plain', useOffse=False)
# plt.show()


# benchmark.plot_returns((all_holdings*assets).sum(axis='columns'), [datetime(1871,1,1),datetime(2023,2,10)], False)
