import pandas as pd
from pathlib import Path
df = pd.read_csv("data/raw/customer_monthly.csv", parse_dates=["month"])
out_root = Path("data/lake/financial"); out_root.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_root / "customer_monthly.parquet", index=False)
print("Wrote", out_root / "customer_monthly.parquet")
