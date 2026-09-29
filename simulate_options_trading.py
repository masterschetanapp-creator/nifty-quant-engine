import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "walkforward_131_months_master.csv"))

print("="*65)
print("131-MONTH OPTIONS TRADING SIMULATION (Rs. 1,000 PER MONTH)")
print("Period: 29-Oct-2015 to 29-Sep-2026 (131 Months)")
print("="*65)

# Strategy:
# Every month, on the 29th:
# - Budget = Rs. 1,000
# - If Model predicts Up (Predicted_% > 0): Buy ATM Call Option
# - If Model predicts Down (Predicted_% < 0): Buy ATM Put Option
# At monthly expiry (approx 30 days later):
# - Call Option Payoff = max(0, Actual_% - 0) / Premium_% * Rs. 1,000
# - Put Option Payoff  = max(0, 0 - Actual_%) / Premium_% * Rs. 1,000

for prem_pct in [1.8, 2.0, 2.2, 2.5]:
    total_spent = len(df) * 1000.0 # 131 * 1,000 = Rs. 131,000
    payouts = []
    trades = []
    
    for i, row in df.iterrows():
        act_ret = row["Actual_%"]
        pred_ret = row["Predicted_%"]
        origin_date = row["Origin_Date"]
        target_date = row["Target_Date"]
        
        if pred_ret >= 0:
            trade_type = "BUY CALL"
            payout = max(0.0, act_ret) / (prem_pct) * 1000.0
        else:
            trade_type = "BUY PUT"
            payout = max(0.0, -act_ret) / (prem_pct) * 1000.0
            
        profit = payout - 1000.0
        trades.append({
            "Month": row["Month"],
            "Origin_Date": origin_date,
            "Target_Date": target_date,
            "Trade": trade_type,
            "Predicted_%": pred_ret,
            "Actual_%": act_ret,
            "Cost": 1000.0,
            "Payout": payout,
            "Profit": profit
        })
        payouts.append(payout)
        
    df_trades = pd.DataFrame(trades)
    total_payout = df_trades["Payout"].sum()
    net_profit = total_payout - total_spent
    roi = (net_profit / total_spent) * 100.0
    
    winning_trades = df_trades[df_trades["Profit"] > 0]
    total_loss_trades = df_trades[df_trades["Payout"] == 0]
    partial_trades = df_trades[(df_trades["Payout"] > 0) & (df_trades["Profit"] <= 0)]
    
    print(f"\n--- Model Results at ATM Premium = {prem_pct:.1f}% ---")
    print(f"Total Invested:       Rs. {total_spent:,.0f} (Rs. 1,000 x 131 months)")
    print(f"Total Cash Received:  Rs. {total_payout:,.0f}")
    print(f"Net Profit / Gain:    Rs. {net_profit:+,.0f}")
    print(f"Return on Capital:    {roi:+.1f}%")
    print(f"Winning Months:       {len(winning_trades)} of 131 ({len(winning_trades)/131*100:.1f}%)")
    print(f"Total Loss Months:    {len(total_loss_trades)} of 131 ({len(total_loss_trades)/131*100:.1f}%)")
    print(f"Best Single Month:    Rs. {df_trades['Profit'].max():+,.0f} ({df_trades.loc[df_trades['Profit'].idxmax()]['Trade']} on {df_trades.loc[df_trades['Profit'].idxmax()]['Origin_Date']})")

print("\n" + "="*65)
