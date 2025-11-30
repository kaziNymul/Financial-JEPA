import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)
N = 3000; T = 24
rows = []
for i in range(N):
    income = rng.normal(3000, 500); balance = rng.normal(1000, 300); delin = 0
    spend_trend = rng.normal(0, 10)
    for t in range(T):
        spend = max(0, 800 + 50*np.sin(2*np.pi*t/12) + spend_trend*t + rng.normal(0,60))
        pay_ratio = float(np.clip(rng.normal(0.35, 0.1), 0.0, 1.0))
        payment = spend * pay_ratio
        balance = max(0, balance + spend - payment + rng.normal(0, 30))
        if rng.random() < 0.03: delin = min(6, delin + 1)
        else: delin = max(0, delin - (1 if rng.random()<0.4 else 0))
        rows.append({
            "customer_id": f"C{i:05d}",
            "month": pd.Timestamp("2022-01-01") + pd.offsets.MonthBegin(t),
            "spend": float(spend),
            "income": float(max(1200, income + rng.normal(0, 100))),
            "balance": float(balance),
            "payment_ratio": float(pay_ratio),
            "delinquency_count": int(delin),
        })
df = pd.DataFrame(rows)
Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/customer_monthly.csv", index=False)
print("Wrote data/raw/customer_monthly.csv with", len(df), "rows.")
