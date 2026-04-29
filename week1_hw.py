import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
market_cap = pd.read_csv('market_cap.csv', index_col=0, parse_dates=True)
adj_close = pd.read_csv('quant - s_dq_adjclose_v2.csv', index_col=0, parse_dates=True)

# Compute Size Factor
size_factor = -np.log(market_cap)
size_factor = size_factor.reindex(adj_close.index, method='ffill')

# Compute 5-day forward returns
forward_returns = adj_close.pct_change(periods=5).shift(-5)

# Align common dates and stocks
common_dates = size_factor.index.intersection(forward_returns.index)
common_stocks = size_factor.columns.intersection(forward_returns.columns)

sf = size_factor.loc[common_dates, common_stocks]
fr = forward_returns.loc[common_dates, common_stocks]

# Resample to monthly frequency
sf_monthly = sf.resample('ME').last()
fr_monthly = fr.resample('ME').last()

# Calculate monthly Spearman IC
monthly_ic = sf_monthly.corrwith(fr_monthly, axis=1, method='spearman')

# Resample IC to 6-month frequency by taking the mean over each 6-month period
six_month_ic = monthly_ic.resample('6ME').mean()

# Calculate IR on 6-month IC values
ir = six_month_ic.mean() / six_month_ic.std()

# Print results
print("First few 6-month IC values:")
print(six_month_ic.head())

print(f"\nInformation Ratio (6-month): {ir:.4f}")

# Plot
six_month_ic.plot(marker='o', title='6-Month Spearman IC Over Time')
plt.xlabel("Date")
plt.ylabel("Information Coefficient (IC)")
plt.grid(True)
plt.tight_layout()
plt.show()
