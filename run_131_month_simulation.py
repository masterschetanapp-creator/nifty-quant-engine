import os
import calendar
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "nifty_daily_2015_2026.csv"), parse_dates=["date"]).set_index("date")
dates = df.index
close = df["close"].values

# Extract the 132 monthly anchor dates (29th of each month from Oct 2015 to Sep 2026)
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

actual_returns = np.array([((origins[i+1][2] / origins[i][2]) - 1.0) * 100.0 for i in range(len(origins)-1)])

# Precalculate features for all 131 months
features = []
for i in range(len(origins) - 1):
    o_date, o_pos, o_close = origins[i]
    c = close[:o_pos+1]
    dd252 = (o_close / np.max(c[-252:]) - 1.0) * 100.0
    mom120 = (o_close / c[-120] - 1.0) * 100.0
    mom20 = (o_close / c[-20] - 1.0) * 100.0
    sma50 = c[-50:].mean()
    features.append({
        "month": i + 1,
        "o_date": o_date,
        "t_date": origins[i+1][0],
        "o_close": o_close,
        "t_close": origins[i+1][2],
        "dd252": dd252,
        "mom120": mom120,
        "mom20": mom20,
        "sma50": sma50,
        "act_pct": actual_returns[i]
    })

def simulate_series(params, feats):
    w20, w120, w_dd, th_rev, th_trap, d_rev, d_trap, d_exp, lr = params
    adaptive_bias = 0.0
    errs, preds, regimes, hits, in_80s = [], [], [], [], []
    
    # 80% confidence band sigma
    sigma_m = 3.65
    
    for f in feats:
        if f["dd252"] < th_rev:
            regime = "Value Reversal & Bounce"
            base_drift = d_rev
        elif f["mom120"] > th_trap and f["dd252"] > -3.0:
            regime = "Valuation Trap & Snapback"
            base_drift = d_trap
        elif f["o_close"] < f["sma50"] and f["mom20"] < -1.0:
            regime = "Tactical Pullback"
            base_drift = -0.50
        else:
            regime = "Structural Expansion & SIP Floor"
            base_drift = d_exp
            
        pred_pct = base_drift + w20 * (f["mom20"] * 0.1) + w120 * (f["mom120"] * 0.05) + w_dd * (f["dd252"] * 0.05) - adaptive_bias
        act_pct = f["act_pct"]
        err = act_pct - pred_pct
        abs_err = abs(err)
        
        hit = (act_pct > 0) == (pred_pct > 0)
        p10 = f["o_close"] * (1.0 + (pred_pct - 1.28 * sigma_m) / 100.0)
        p90 = f["o_close"] * (1.0 + (pred_pct + 1.28 * sigma_m) / 100.0)
        in_80 = (p10 <= f["t_close"] <= p90)
        
        errs.append(abs_err)
        preds.append(pred_pct)
        regimes.append(regime)
        hits.append(hit)
        in_80s.append(in_80)
        
        # Adaptive bias update
        adaptive_bias = (1.0 - lr) * adaptive_bias + lr * err
        
    return np.array(errs), np.array(preds), regimes, np.array(hits), np.array(in_80s)

# Test Champion 1,000-run parameters on the full 131 months
from champion_1000_params import champion_params

errs_champ, preds_champ, regimes_champ, hits_champ, in80_champ = simulate_series(champion_params, features)

print("="*70)
print("CHAMPION MODEL TESTED ON EXTENDED 131-MONTH DATASET (2015-2026):")
print(f"Total Completed Cycles:    {len(errs_champ)} Months")
print(f"Overall MAE:               {errs_champ.mean():.3f} pp")
print(f"Overall RMSE:              {np.sqrt((errs_champ**2).mean()):.3f} pp")
print(f"Directional Hit Rate:      {hits_champ.mean()*100:.1f}%")
print(f"80% Confidence Band:       {in80_champ.mean()*100:.1f}%")
corr_full = np.corrcoef(actual_returns, preds_champ)[0, 1]
print(f"Predictive Correlation:    {corr_full:.3f}")
print("="*70)

# Breakdown: Early Period (2015-2021, 72 months) vs Modern Period (2021-2026, 59 months)
errs_early = errs_champ[:72]
hits_early = hits_champ[:72]
errs_modern = errs_champ[72:]
hits_modern = hits_champ[72:]

print("\nPeriod Breakdown:")
print(f"Early Era (2015-2021, 72 mo - Demonetization, IL&FS, COVID):")
print(f"  MAE: {errs_early.mean():.3f} pp | Hit Rate: {hits_early.mean()*100:.1f}%")
print(f"Modern Era (2021-2026, 59 mo - Post-COVID SIP Era):")
print(f"  MAE: {errs_modern.mean():.3f} pp | Hit Rate: {hits_modern.mean()*100:.1f}%")
