import pandas as pd
from pathlib import Path

RAW = "data/raw/amex/train_data.csv"
OUT = Path("data/processed/amex"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(RAW) if RAW.endswith(".parquet") else pd.read_csv(RAW)
df = df.rename(columns={"customer_ID":"entity_id","S_2":"timestamp"})

keep_feats = ["B_1","B_2","B_3","B_4","B_5","D_39","D_41","P_2","R_1","R_2","S_3","S_5"]
base = ["entity_id","timestamp"]
num_cols = [c for c in keep_feats if c in df.columns]
df = df[base + num_cols].sort_values(["entity_id","timestamp"])

if "B_11" in df.columns and "P_2" in df.columns:
    df["utilization"] = df["P_2"]/(df["B_11"]+1e-6)
    if "utilization" not in num_cols: num_cols.append("utilization")

for eid, g in df.groupby("entity_id", sort=False):
    g[base + num_cols].to_csv(OUT / f"{eid}.csv", index=False)
print("Done →", OUT, "features:", num_cols)
