import yfinance as yf
import pandas as pd
import numpy as np
import calendar
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Fetching full NIFTY 50 daily history (2015-2026) from Yahoo Finance...")

# Download ^NSEI daily bars
df = yf.Ticker("^NSEI").history(start="2015-01-01", end="2026-09-30")
df.index = pd.to_datetime(df.index).tz_localize(None)
df = df[["Close", "Volume", "High", "Low", "Open"]].dropna()
df.rename(columns={"Close": "close", "Volume": "volume", "High": "high", "Low": "low", "Open": "open"}, inplace=True)
df.index.name = "date"
df.reset_index(inplace=True)

out_path = os.path.join(BASE_DIR, "nifty_daily_2015_2026.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} daily trading records to {out_path}")
print("Date range:", df["date"].min().strftime("%Y-%m-%d"), "to", df["date"].max().strftime("%Y-%m-%d"))

# Extract monthly origins: 29th of each month from 2015-10 to 2026-09
dates = df["date"].values
close = df["close"].values

origins = []
cur_y, cur_m = 2015, 10
for i in range(150):
    max_d = calendar.monthrange(cur_y, cur_m)[1]
    target_d = min(29, max_d)
    m_str = f"{cur_y:04d}-{cur_m:02d}-{target_d:02d}"
    pos = int(np.searchsorted(dates, np.datetime64(m_str), side="right") - 1)
    if pos < len(dates):
        if not origins or origins[-1][1] != pos:
            origins.append((pd.Timestamp(dates[pos]), pos, float(close[pos])))
    cur_m += 1
    if cur_m > 12:
        cur_m = 1
        cur_y += 1
    if cur_y == 2026 and cur_m > 9:
        break

print(f"Total monthly intervals: {len(origins)} anchor points -> {len(origins)-1} monthly forecast cycles!")
print(f"Start: {origins[0][0].strftime('%Y-%m-%d')} (Close: {origins[0][2]:,.2f})")
print(f"End:   {origins[-1][0].strftime('%Y-%m-%d')} (Close: {origins[-1][2]:,.2f})")
