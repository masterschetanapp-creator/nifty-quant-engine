import pandas as pd
import numpy as np
import os
import calendar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "nifty_daily_2015_2026.csv"), parse_dates=["date"]).set_index("date")
dates = df.index
close = df["close"].values

origins = []
cur_y, cur_m = 2015, 10
for i in range(150):
    max_d = calendar.monthrange(cur_y, cur_m)[1]
    target_d = min(29, max_d)
    m_str = f"{cur_y:04d}-{cur_m:02d}-{target_d:02d}"
    pos = int(dates.searchsorted(pd.Timestamp(m_str), side="right") - 1)
    if pos < len(dates):
        if not origins or origins[-1][1] != pos:
            origins.append((dates[pos], pos, float(close[pos])))
    cur_m += 1
    if cur_m > 12:
        cur_m = 1
        cur_y += 1
    if cur_y == 2026 and cur_m > 9:
        break

returns = np.array([((origins[i+1][2] / origins[i][2]) - 1.0) * 100.0 for i in range(len(origins)-1)])

print("131 Months Return Statistics (Oct 2015 to Sep 2026):")
print(f"Total cycles:    {len(returns)}")
print(f"Mean return:     {returns.mean():.2f}%")
print(f"Std deviation:   {returns.std():.2f}%")
print(f"Min return:      {returns.min():.2f}%")
print(f"Max return:      {returns.max():.2f}%")

min_i = int(np.argmin(returns))
max_i = int(np.argmax(returns))
w_start = origins[min_i][0].strftime('%Y-%m-%d')
w_end = origins[min_i+1][0].strftime('%Y-%m-%d')
b_start = origins[max_i][0].strftime('%Y-%m-%d')
b_end = origins[max_i+1][0].strftime('%Y-%m-%d')

print(f"Worst month:     {w_start} to {w_end} -> {returns[min_i]:.2f}% (COVID Crash)")
print(f"Best month:      {b_start} to {b_end} -> {returns[max_i]:.2f}% (Post-COVID Recovery)")

up_m = int((returns > 0).sum())
print(f"Up months:       {up_m} ({up_m/len(returns)*100:.1f}%)")
print(f"Down months:     {len(returns)-up_m} ({(len(returns)-up_m)/len(returns)*100:.1f}%)")
