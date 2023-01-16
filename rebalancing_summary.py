# rebalancing portfolio assets according to asset allocation plans, or 'weights' in this code
for yearend in yearly_prices.index:    
    if yearend == yearly_prices.index[0]:
        continue
    add = pd.DataFrame(yearly_prices.loc[yearend] * holdings.iloc[-1,:], columns=[yearend]).T
    values = pd.concat([values, add])
    
    add = values.loc[yearend]/yearly_prices.loc[yearend]
    holdings = pd.concat([holdings, add.round(decimals=1).astype('int').to_frame().T])
    
    # The rebalancing algorithm is as follow:
    # 1. find off values = calculate prices lower or higher than allocation plans
    #   off values = yearend prices - yearend prices * allocation ratio
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
