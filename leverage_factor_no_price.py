
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
assets = pd.read_csv('total_assets.csv', index_col=0, parse_dates=True)
equity = pd.read_csv('equity.csv', index_col=0, parse_dates=True)
announcement = pd.read_csv('announcement_date.csv', index_col=0)
market_cap = pd.read_csv('market_cap.csv', index_col=0, parse_dates=True)

# Preprocess
assets.index = assets.index.to_period('M').to_timestamp('M')
equity.index = equity.index.to_period('M').to_timestamp('M')
market_cap.index = market_cap.index.to_period('M').to_timestamp('M')
announcement.index = pd.to_datetime(announcement.index, errors='coerce')
announcement = announcement.applymap(lambda x: pd.to_datetime(x, errors='coerce'))

assets = assets[~assets.index.duplicated(keep='last')]
equity = equity[~equity.index.duplicated(keep='last')]
assets = assets.reindex(market_cap.index).ffill()
equity = equity.reindex(market_cap.index).ffill()

# Leverage Factor
leverage = assets / equity
leverage_factor = leverage.applymap(lambda x: np.log(x) if pd.notnull(x) and x > 0 else np.nan)

# Align with announcement dates
def _deal_ann_date(anndt_bs, trade_days, stocks):
    matric_anndate_bs = {}
    for keyday in trade_days:
        matric_anndate_bs[keyday] = {}
        for ticker in stocks:
            try:
                bs = anndt_bs[ticker]
            except:
                continue
            report_date = -1
            for k, ann_dt in bs.items():
                if pd.notnull(ann_dt):
                    if ann_dt >= keyday:
                        break
                    else:
                        report_date = int(k.strftime('%Y%m%d'))
            matric_anndate_bs[keyday][ticker] = report_date
    return pd.DataFrame(matric_anndate_bs).T

def _date_adjust(df, matric_anndate):
    data = {}
    for c in df.columns:
        try:
            xd = matric_anndate[c]
            xd = xd.loc[xd > 0]
        except:
            continue
        valid_dates = pd.to_datetime(xd.values.astype(str), format='%Y%m%d', errors='coerce')
        valid_dates = valid_dates[valid_dates.isin(df.index)]
        if len(valid_dates) == 0:
            continue
        v = df.loc[valid_dates, c].values
        t = xd.index[:len(v)]
        data[c] = pd.Series(v, index=t)
    return pd.DataFrame(data)

matric_anndate = _deal_ann_date(announcement, market_cap.index, market_cap.columns)
leverage_factor_td = _date_adjust(leverage_factor, matric_anndate)

# Note: No backtest due to lack of price data
# Instead, describe the factor and plot its distribution
mean_exposure = leverage_factor_td.mean(axis=1)
mean_exposure.plot(title='Average Leverage Factor Over Time', figsize=(10, 4))
plt.ylabel('Mean log(Leverage)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Distribution on most recent date
latest = leverage_factor_td.iloc[-1].dropna()
latest.hist(bins=50, figsize=(8, 4))
plt.title('Cross-Sectional Distribution of Leverage (Latest Date)')
plt.xlabel('log(Leverage)')
plt.grid(True)
plt.tight_layout()
plt.show()
