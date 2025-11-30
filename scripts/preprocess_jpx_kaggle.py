import pandas as pd
from pathlib import Path

RAW = "data/raw/jpx/train_files/stock_prices.csv"
OUT = Path("data/processed/jpx"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
df = df.rename(columns={"SecuritiesCode":"entity_id","Date":"timestamp"})
df = df.sort_values(["entity_id","timestamp"])

df["ret_1d"]  = df.groupby("entity_id")["Close"].pct_change()
df["ret_5d"]  = df.groupby("entity_id")["Close"].pct_change(5)
df["vol_5d"]  = df.groupby("entity_id")["Close"].rolling(5).std().reset_index(0,drop=True)
df["atr_14d"] = (df["High"]-df["Low"]).rolling(14).mean()

keep = ["entity_id","timestamp","Open","High","Low","Close","Volume","ret_1d","ret_5d","vol_5d","atr_14d"]
df = df[keep]

for eid, g in df.groupby("entity_id", sort=False):
    g.to_csv(OUT / f"{eid}.csv", index=False)
print("Done →", OUT, "features:", keep[2:])
