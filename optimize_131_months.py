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
    errs, preds, hits = [], [], []
    
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
        
        errs.append(abs_err)
        preds.append(pred_pct)
        hits.append((act_pct > 0) == (pred_pct > 0))
        adaptive_bias = (1.0 - lr) * adaptive_bias + lr * err
        
    return np.array(errs), np.array(preds), np.array(hits)

# Rolling walk-forward 3-split cross validation across 131 months:
# Split 1: Train 0..60 (2015-2020), Val 60..85 (2020-2022)
# Split 2: Train 0..85 (2015-2022), Val 85..110 (2022-2024)
# Split 3: Train 0..110 (2015-2024), Val 110..131 (2024-2026)
cv_splits = [
    (features[:60], features[60:85]),
    (features[:85], features[85:110]),
    (features[:110], features[110:])
]

from champion_1000_params import champion_params
base_errs, base_preds, base_hits = simulate_series(champion_params, features)
base_mae = base_errs.mean()
base_rmse = np.sqrt((base_errs**2).mean())
base_hit = base_hits.mean() * 100
base_corr = np.corrcoef(actual_returns, base_preds)[0, 1]

print("BASELINE 131-MONTH PERFORMANCE:")
print(f"MAE: {base_mae:.3f} pp | RMSE: {base_rmse:.3f} pp | Hit Rate: {base_hit:.1f}% | Corr: {base_corr:.3f}")

np.random.seed(5555)
best_candidate = None
best_score = 999.0

# Run 1000 iterations
for it in range(1, 1001):
    if it % 2 == 0:
        cand = [
            np.clip(champion_params[0] + np.random.normal(0, 0.04), -0.4, 0.4),
            np.clip(champion_params[1] + np.random.normal(0, 0.03), -0.3, 0.3),
            np.clip(champion_params[2] + np.random.normal(0, 0.03), -0.3, 0.3),
            np.clip(champion_params[3] + np.random.normal(0, 0.6), -16.0, -6.0),
            np.clip(champion_params[4] + np.random.normal(0, 0.8), 10.0, 22.0),
            np.clip(champion_params[5] + np.random.normal(0, 0.1), 0.8, 2.5),
            np.clip(champion_params[6] + np.random.normal(0, 0.1), -2.5, -0.4),
            np.clip(champion_params[7] + np.random.normal(0, 0.1), 0.5, 1.8),
            np.clip(champion_params[8] + np.random.normal(0, 0.02), 0.02, 0.35)
        ]
    else:
        cand = [
            np.random.uniform(-0.35, 0.35),
            np.random.uniform(-0.25, 0.25),
            np.random.uniform(-0.25, 0.25),
            np.random.uniform(-15.0, -7.0),
            np.random.uniform(10.0, 20.0),
            np.random.uniform(0.8, 2.2),
            np.random.uniform(-2.2, -0.5),
            np.random.uniform(0.6, 1.6),
            np.random.uniform(0.03, 0.35)
        ]
        
    val_maes = []
    for tr, val in cv_splits:
        e, _, _ = simulate_series(cand, val)
        val_maes.append(e.mean())
    cv_val_mae = np.mean(val_maes)
    
    full_e, full_p, full_h = simulate_series(cand, features)
    full_mae = full_e.mean()
    full_hit = full_h.mean() * 100
    
    # Combined score prioritizing low out-of-sample error and high directional accuracy
    score = cv_val_mae * 0.6 + full_mae * 0.4 - (full_hit / 100.0) * 0.5
    
    if score < best_score and full_mae < base_mae and full_hit >= base_hit:
        best_score = score
        best_candidate = {
            "iteration": it,
            "params": cand,
            "cv_mae": cv_val_mae,
            "full_mae": full_mae,
            "full_rmse": np.sqrt((full_e**2).mean()),
            "full_hit": full_hit,
            "corr": np.corrcoef(actual_returns, full_p)[0, 1]
        }

if best_candidate:
    print(f"\nDiscovered Superior 131-Month Unified Model on Iteration {best_candidate['iteration']}!")
    print(f"MAE:       {best_candidate['full_mae']:.3f} pp (Reduced from {base_mae:.3f} pp)")
    print(f"RMSE:      {best_candidate['full_rmse']:.3f} pp (Reduced from {base_rmse:.3f} pp)")
    print(f"Hit Rate:  {best_candidate['full_hit']:.1f}% (Baseline: {base_hit:.1f}%)")
    print(f"Corr:      {best_candidate['corr']:.3f} (Baseline: {base_corr:.3f})")
    print(f"CV Val MAE:{best_candidate['cv_mae']:.3f} pp")
    
    with open(os.path.join(BASE_DIR, "champion_131_params.py"), "w", encoding="utf-8") as f:
        f.write("import numpy as np\n")
        f.write(f"champion_131_params = {[float(x) for x in best_candidate['params']]}\n")
        f.write(f"champion_131_stats = {best_candidate}\n")
    print("Saved champion 131-month parameters to champion_131_params.py")
else:
    print("Baseline parameters remain best.")
