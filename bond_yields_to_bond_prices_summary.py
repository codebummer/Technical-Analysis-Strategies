# bond invest amount
bonds_invest = 10_000

# price at the time of purchase = invest amount * (1 + yield rate)
# asset is a pandas.Series
fixed = bonds_invest * (assets['US10Y'].values[0]/100+1)

# prices in the sequential time frame = buy price * (1 + change rates in yields * -1)
# -1 is multiplied because bond yields and prices are in a negative relationship, i.e. 1% increase in yiels means proportional change in its price in a negative way
# 1 is added to calculate principal and interests at the same time
prices = fixed * assets['US10Y'].pct_change().multiply(-1).add(1)

# pandas.Series.pct_change() will empty the first row (which represents the purchase price), which means the buy price should be filled in that place.
prices[0] = fixed

# now you got bond prices pandas.Series
prices
