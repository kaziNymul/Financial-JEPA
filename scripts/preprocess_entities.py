import pandas as pd
from pathlib import Path

IN = "data/lake/financial/customer_monthly.parquet"
OUT = Path("data/processed/bank"); OUT.mkdir(parents=True, exist_ok=True)

keep_feats = ["spend","income","balance","payment_ratio","delinquency_count"]
df = pd.read_parquet(IN)
df = df.rename(columns={"customer_id": "entity_id", "month": "timestamp"})
df = df.sort_values(["entity_id", "timestamp"])
df = df[["entity_id","timestamp"] + keep_feats]

for eid, g in df.groupby("entity_id", sort=False):
    g.to_csv(OUT / f"{eid}.csv", index=False)
print("Wrote", len(list(OUT.glob("*.csv"))), "entity files to", OUT)
