import yfinance as yf
import pandas as pd
import numpy as np
import os
import calendar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Fetching complete GOLDBEES daily data from 2015 to 2026...")

df_gold = yf.Ticker("GOLDBEES.NS").history(start="2015-01-01", end="2026-09-30")
df_gold.index = pd.to_datetime(df_gold.index).tz_localize(None)
df_gold = df_gold[["Close", "Volume", "High", "Low", "Open"]].dropna()
df_gold.rename(columns={"Close": "close", "Volume": "volume", "High": "high", "Low": "low", "Open": "open"}, inplace=True)
df_gold.index.name = "date"
df_gold.reset_index(inplace=True)

out_csv = os.path.join(BASE_DIR, "goldbees_daily_2015_2026.csv")
df_gold.to_csv(out_csv, index=False)
print(f"Saved {len(df_gold)} trading days to {out_csv}")
print("Date range:", df_gold["date"].min().strftime("%Y-%m-%d"), "to", df_gold["date"].max().strftime("%Y-%m-%d"))

# Extract 132 monthly anchors on the 29th of each month from Oct 2015 to Sep 2026
dates = df_gold["date"].values
close = df_gold["close"].values

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

print(f"Total monthly intervals found for Gold: {len(origins)} anchors -> {len(origins)-1} monthly cycles!")
print(f"First anchor: {origins[0][0].strftime('%Y-%m-%d')} at Rs. {origins[0][2]:.2f}")
print(f"Last anchor:  {origins[-1][0].strftime('%Y-%m-%d')} at Rs. {origins[-1][2]:.2f}")

returns = np.array([((origins[i+1][2] / origins[i][2]) - 1.0) * 100.0 for i in range(len(origins)-1)])
print("\n131-Month Gold Monthly Return Statistics:")
print(f"Mean Monthly Return: {returns.mean():.2f}%")
print(f"Std Deviation:       {returns.std():.2f}%")
print(f"Min Monthly Return:  {returns.min():.2f}%")
print(f"Max Monthly Return:  {returns.max():.2f}%")
up_months = (returns > 0).sum()
print(f"Up Months:           {up_months} of {len(returns)} ({(up_months/len(returns))*100:.1f}%)")
