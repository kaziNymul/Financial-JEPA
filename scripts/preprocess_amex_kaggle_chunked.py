import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

RAW = "data/raw/amex/train_data.csv"      # your downloaded file
OUT = Path("data/processed/amex")
OUT.mkdir(parents=True, exist_ok=True)

# Keep a SMALL feature set to fit on CPU (expand later if you want)
# These exist in AMEX; script will ignore missing ones.
KEEP = [
    "customer_ID","S_2",                # id + timestamp
    "B_1","B_2","B_3","B_4","B_5",
    "D_39","D_41",
    "P_2",
    "R_1","R_2",
    "S_3","S_5",
]

DTYPES = defaultdict(lambda: "float32")
DTYPES.update({
    "customer_ID": "string",
    "S_2": "string",  # parse to datetime later
})

# How many rows to read per chunk (tune for your RAM; 1e6 is ~large)
CHUNK = 400_000

def write_group(df):
    """Append one customer's rows to its CSV (sorted by time)."""
    cid = df["customer_ID"].iloc[0]
    f = OUT / f"{cid}.csv"
    # We only append (we’ll sort per-file after the full pass if needed)
    mode = "a" if f.exists() else "w"
    header = not f.exists()
    df.to_csv(f, index=False, mode=mode, header=header)

def main():
    used_cols = [c for c in KEEP if c]   # only those we want
    # Use low_memory=False to avoid dtype guessing, set chunksize
    reader = pd.read_csv(
        RAW,
        usecols=lambda c: c in used_cols,
        dtype=DTYPES,
        chunksize=CHUNK,
        low_memory=False
    )
    n_rows = 0
    for i, chunk in enumerate(reader, 1):
        # Parse date and sort within chunk for stable append
        chunk["timestamp"] = pd.to_datetime(chunk["S_2"], errors="coerce")
        cols = ["customer_ID","timestamp"] + [c for c in used_cols if c not in ("customer_ID","S_2")]
        chunk = chunk[cols].dropna(subset=["customer_ID","timestamp"])
        # Group inside the chunk and append to per-customer files
        for cid, g in chunk.groupby("customer_ID", sort=False):
            write_group(g)
        n_rows += len(chunk)
        if i % 10 == 0:
            print(f"[chunk {i}] processed rows so far: {n_rows:,}")

if __name__ == "__main__":
    main()
