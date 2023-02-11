import math
from benchmark_datareader import Benchmark

benchmark = Benchmark()


# make price matrix and weight matrix of the portfolio

# initial amount of investment
invested = 30_000
# weights for assets in ratio
weights = pd.Series({'S&P500':0.3, 'US10Y':0.5, 'XAU/USD':0.15, 'USD':0.05}, name='Weights')
# initialize holdings that represents stock numbers held, following the ratio
holdings = {}
for asset in assets.keys():
    if asset == 'S&P500':
        holdings[asset] = math.floor(invested*weights[asset]/(assets[asset].values[0]*0.1)) # VOO is approximately 9.1% of S&P500
    elif asset == 'US10Y':        
        holdings[asset] = math.floor(invested*weights[asset]/10) # Minimum amount to buy US10Y is USD 10
    elif asset == 'XAU/USD':
        holdings[asset] = math.floor(invested*weights[asset]/(assets[asset].values[0]*0.1)) # GLD is approximately 9.3% of XAU/USD
    else:
        holdings[asset] = math.floor(invested*weights[asset]/assets[asset].values[0])
holdings = pd.Series(holdings, name='Holdings')

bonds_invest = 10000
fixed = bonds_invest * (assets['US10Y'].values[0]/100+1)
prices = fixed * assets['US10Y'].pct_change().multiply(-1).add(1)
prices[0] = fixed
prices

# convert bond yields to bonds prices
def calc_bond_price(assgined_amount, yields):
    '''
    aggsigned_amount: initial cash amount assigned to bonds in the working investment strategy,
    yields: a pandas series or dataframe that holds daily bond yields
    '''
    # calculate the inital fixed income amount from the initial cash amount assigned to invest in bonds
    # 
    # bonds yields amount (or fixed income amount) at the time of purchase: fixed_income (in the below code block)
    # = bond yield at the time of purchase (in decimal digits, not in percent) + 1(to include the principal) * the initial cash amount to invest in bonds
    fixed_income = assgined_amount * (yields.values[0]/100+1)
    # fixed_income = assgined_amount * (yields[0]/100+1) # In case getting yields as a numpy array

    # the later market prices of the above fixed income amount at the time of purchase
    # = [bond yield change (in decimal digits, not in percent)
    # * -1(to apply the negative relationship between bond prices and bond yields) 
    # + 1(to include the principal) ] * the initial fixed income amount at the time of purchase
    # Note: pandas.DataFrame.pct_change() does not generate changes in percent, but in decimal digits.
    bond_prices = yields.pct_change().multiply(-1).add(1) * fixed_income
    # bond_prices = (np.diff(yields)/yields[:-1]*-1+1) * np.array([fixed_income for _ in range(len(yields)-1)]) # In case getting yields as a numpy array
    
    # fill the first row which got empty after pct_change() with the initial fixed income amount at the time of purchase
    bond_prices[0] = fixed_income
    # bond_prices = np.append(fixed_income, bond_prices) # In case getting yields as a numpy array
    return bond_prices

# make sure you don't loop through the entire list of (assets.index.year)
# just loop through list(set(assets.index.year))
bond_prices = pd.Series()
for year in tqdm(list(set(assets.index.year))):
    add = calc_bond_price(invested*weights['US10Y'], assets['US10Y'].groupby(assets.index.year).get_group(year))
    bond_prices = pd.concat([bond_prices, add])

# alternative way to implement the above. Also expensive
bond_prices = pd.Series()
for year in tqdm(list(set(assets.index.year))):
    add = calc_bond_price(invested*weights['US10Y'], assets['US10Y'].loc[assets.index.year==year])
    bond_prices = pd.concat([bond_prices, add])

prices = pd.concat([assets['S&P500'], bond_prices], axis='columns')
prices.columns = ['S&P500','US10Y']

# bond prices when annually reinvested with past years' yearend prices 
bond_prices_reinvested = pd.Series()
bond_reinvest = invested * weights['US10Y']
for year in tqdm(list(set(assets.index.year))):
    add = calc_bond_price(bond_reinvest, assets['US10Y'].groupby(assets.index.year).get_group(year))
    bond_prices_reinvested = pd.concat([bond_prices_reinvested, add])
    bond_reinvest = bond_prices_reinvested[-1]

bond_returns = bond_prices.pct_change().add(1).cumprod()
bond_returns_reinvested = bond_prices_reinvested.pct_change().add(1).cumprod()
sns.lineplot(bond_returns)
sns.lineplot(bond_returns_reinvested)
plt.legend(['BOND RETURNS', 'BOND RETURNS REINVESTED'])
plt.show()

sp500 = assets['S&P500'].groupby(assets.index.year).get_group(2020).pct_change().add(1).cumprod()

min_date = lambda x: x.index[x==x.min()]
max_date = lambda x: x.index[x==x.max()]
best = max_date(bond_prices)
worst = min_date(bond_prices)
bond_prices.loc[best]
bond_prices.loc[worst]
bond_prices.loc[datetime(2020,3,2):datetime(2020, 3, 30)]
assets.loc[worst]
assets.loc[best]
y2020 = bond_prices[bond_prices.index.year==2020]
us10y = y2020.pct_change().add(1).cumprod()
us10y202003 = y2020.groupby(y2020.index.month).get_group(3).pct_change().add(1).cumprod()

def plot_returns(assets, dates, cumul=True):
    '''
    assets: a dataframe that contains assets data to plot
    dates: a list of datetime values that point to start and end dates. Both values should be idential in assets.index values. use [datetime(start date), datetime(end date)]
    cumul: If True, plots cumulative product of changes in prices. If False, plots just changes in prices.
    '''
    if cumul == True:
        for asset in assets.columns:
            if asset == 'US10Y':
                sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change().multiply(-1).add(1).cumprod())
            else:
                sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change().add(1).cumprod())
    else:
        for asset in assets.columns:
            if asset == 'US10Y':
                sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change().multiply(-1))
            else:
                sns.lineplot(assets[asset].loc[dates[0]:dates[1]].pct_change())
    plt.legend(labels=assets)
    plt.show()

plot_returns(assets[['S&P500', 'US10Y']], [datetime(2020,1,2),datetime(2020,12,31)], cumul=False)
assets.groupby(assets.index.year).get_group(2020)        





holdings*assets.iloc[0,:].sum()
prices=holdings*assets.iloc[0,:]
prices/prices.sum()-weights

prices = assets.iloc[0,:]
values = holdings*prices
values['US10Y'] = holdings['US10Y']*10
values/values.sum()-weights

holdings


for yearend in yearly_prices.index:    
    if yearend == yearly_prices.index[0]:
        continue
    add = pd.DataFrame(yearly_prices.loc[yearend] * holdings.iloc[-1,:], columns=[yearend]).T
    values = pd.concat([values, add])
    
    add = values.loc[yearend]/yearly_prices.loc[yearend]
    holdings = pd.concat([holdings, add.round(decimals=1).astype('int').to_frame().T])
    
    # The rebalancing algorithm is as follow:
    # 1. find off values = calculate prices lower or higher than allocation plans
    #   off values = yearend prices - sum of yearend prices * allocation ratio
    off_values = values.loc[yearend] - weights/100*values.loc[yearend].sum()
    
    # 2. find off quantities = calculate asset quantities lower or higher than allocation plans, based on off values
    #   off quantities = off values / year end prices
    #   round the above off quantities
    #   convert them to integer
    off_qty = (off_values/yearly_prices.loc[yearend]).round(decimals=1).astype('int')   
    
    
    # 3. find gains = calculate money amounts of assets higher than allocation plans, based on off quantities
    #   make all minus off quantities 0 = use pandas.clip(lower=0)
    #   dataframe.clip(lower=0) will keep all elements and prevent NaNs for matrix operations
    #   dataframe.loc[year,off_qty<0] will remove all elements that do not match the condition, 
    #   which will result NaNs and NaNs will cause errors for matrix operations.
    #   The following is an example of the bad practice described above
    #   gains = off_qty.loc[off_qty>0] * yearly_prices.loc[yearend,off_qty>0] 
    gains = off_qty.clip(lower=0) * yearly_prices.loc[yearend]
    
    # 4. find losses = calculate money amounts of assets lower than allocation plans, based on off quantities
    #   same as gains
    #   The following is an example of the bad practice 
    #   losses = off_qty.loc[off_qty<0] * yearly_prices.loc[yearend,off_qty<0]
    losses = off_qty.clip(upper=0) * yearly_prices.loc[yearend]
    
    # 5. if the total gains are less than a minimum unit price of the loss assets,
    #   skip rebalancing, because not a single minimum asset can be purchased
    #
    #   a. To find the minimum unit price of the loss assets,
    #      find unit prices of loss assets
    #
    #   b. convert nonzero quantities of loss assets to 1: 
    #      off_qty.clip(upper=0).where(off_qty.clip(upper=0)==0,1)
    #   
    #   c. multiply them by yearend prices, which results in loss unit prices
    loss_unit_prices = off_qty.clip(upper=0).where(off_qty.clip(upper=0)==0,1) * yearly_prices.loc[yearend]
    
    #   d. exclude zeros to accurately locate minimum unit price
    #      loss_unit_prices.loc[loss_unit_prices!=0]
    
    #   e. sort loss unit prices that are not zero.
    loss_unit_prices = loss_unit_prices.loc[loss_unit_prices!=0].sort_values()
    #   f. locate the minimum unit price of loss assets
    min_asset, min_unit_price = loss_unit_prices.index[0], loss_unit_prices[0]    
    #   g. execute with the above conditions to skip the rebalancing
    if gains.sum() < min_unit_price:
        continue
    
    # The following is rather inaccurate condition to skip rebalancing
    # because it looks at the minimum amount of the total loss asset,
    # not the minimum amount of the unit loss asset
    # if gains.sum() < losses.min(): 
    #     continue
    

    # 6. find asset quantities to restock
    # quantify asset quantities to restock from money amounts to restock
    #   a. find rebalance ratio 
    #       = find ratio of loss assets' off quantities
    #       off quantities of only loss assets / sum of off quantities of only loss assets   
    rebalance_ratio = off_qty.clip(upper=0) / off_qty.clip(upper=0).sum()
    #   b. find amounts ratio that can be distributed to buy restocking assets after selling gained assets
    #       sum of gains * off quantities ratio of loss assets / yearend prices
    #   c. to find quantities to restock,
    #       round the restock quantitiy ratio above that can be distributed from selling gained assets
    rebalance_qty = (gains.sum()*rebalance_ratio/yearly_prices.loc[yearend]).round(decimals=1).astype('int')
    #   d. find actual prices to restock
    #       quantities to restock * yearend prices
    #   e. sort the restock prices calculated
    rebalance_order = (rebalance_qty.abs()*yearly_prices.loc[yearend]).sort_values(ascending=False)
    #   f. find the threshold to buy restocking assets
    #       calculate cumulative sum of the above sorted restock prices
    #       list restock prices of assets that are less than the total gains
    rebalance_assets = rebalance_order.cumsum() < gains.sum() 

    # 7. correct holdings accoringly
    #   rebalance_qty.loc[rebalance_assets] refers to assets to restock (plus values so add this from holdings)
    #   off_qty.loc[off_qty>0] refers to assets to sell (plus values so subract this from holdings)
    holdings.loc[yearend] = holdings.loc[yearend] + rebalance_qty.loc[rebalance_assets] - off_qty.clip(lower=0)
