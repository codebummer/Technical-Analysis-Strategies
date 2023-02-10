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
