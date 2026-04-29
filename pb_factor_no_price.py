
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
market_cap = pd.read_csv('market_cap.csv', index_col=0, parse_dates=True)
equity = pd.read_csv('equity.csv', index_col=0, parse_dates=True)
announcement = pd.read_csv('announcement_date.csv', index_col=0)

# Preprocess
market_cap.index = market_cap.index.to_period('M').to_timestamp('M')
equity.index = equity.index.to_period('M').to_timestamp('M')
announcement.index = pd.to_datetime(announcement.index, errors='coerce')
announcement = announcement.applymap(lambda x: pd.to_datetime(x, errors='coerce'))

# Clean and align
equity = equity[~equity.index.duplicated(keep='last')]
equity = equity.reindex(market_cap.index).ffill()

# Compute PB and factor (inverse log PB)
pb = market_cap / equity
value_factor = -pb.applymap(lambda x: np.log(x) if pd.notnull(x) and x > 0 else np.nan)

# Align to announcement dates
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
value_factor_td = _date_adjust(value_factor, matric_anndate)

# Visualize average exposure over time
mean_exposure = value_factor_td.mean(axis=1)
mean_exposure.plot(title='Average Inverse PB Factor Over Time', figsize=(10, 4))
plt.ylabel('Mean -log(PB)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Cross-sectional distribution on latest date
latest = value_factor_td.iloc[-1].dropna()
latest.hist(bins=50, figsize=(8, 4))
plt.title('Cross-Sectional Distribution of Inverse PB (Latest Date)')
plt.xlabel('-log(PB)')
plt.grid(True)
plt.tight_layout()
plt.show()
