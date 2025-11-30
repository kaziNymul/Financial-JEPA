#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse

def parse_args():
    ap = argparse.ArgumentParser(description="AMEX → per-customer CSVs (keep ALL features)")
    ap.add_argument("--raw", default="data/raw/amex/train_data.csv",
                    help="Path to AMEX train_data.csv")
    ap.add_argument("--out", default="data/processed/amex",
                    help="Output directory for per-customer CSVs")
    ap.add_argument("--chunk", type=int, default=400_000,
                    help="CSV chunk size (tune for your RAM)")
    ap.add_argument("--time-col", default="S_2",
                    help="Timestamp column in raw (AMEX uses S_2)")
    return ap.parse_args()

def write_group(df: pd.DataFrame, out_dir: Path):
    """Append one customer's rows to its CSV, keeping header only once."""
    cid = df["customer_ID"].iloc[0]
    f = out_dir / f"{cid}.csv"
    mode = "a" if f.exists() else "w"
    header = not f.exists()
    df.to_csv(f, index=False, mode=mode, header=header)

def main():
    args = parse_args()
    RAW = args.raw
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    TIME_COL = args.time_col
    CHUNK = args.chunk

    # 1) Peek header to get ALL columns
    head = pd.read_csv(RAW, nrows=5, low_memory=False)
    all_cols = head.columns.tolist()

    # Keep ALL columns; we will only add a 'timestamp' (parsed) and keep original S_2 for traceability
    usecols = all_cols
    if "customer_ID" not in usecols:
        raise SystemExit("customer_ID column not found in raw file")
    if TIME_COL not in usecols:
        raise SystemExit(f"{TIME_COL} (timestamp) column not found in raw file")

    # Pandas dtype hints: don't force floats here; we keep raw types and let the model loader cast safely later
    # Only ensure identifier & timestamp parse
    dtype_map = defaultdict(lambda: "object")
    dtype_map["customer_ID"] = "string"
    dtype_map[TIME_COL] = "string"  # parse to datetime -> 'timestamp' next

    # 2) Stream through the big CSV
    reader = pd.read_csv(
        RAW,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=CHUNK,
        low_memory=False
    )

    n_rows = 0
    for i, chunk in enumerate(reader, 1):
        # 3) Parse timestamp
        chunk["timestamp"] = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        # Drop rows missing key fields
        chunk = chunk.dropna(subset=["customer_ID", "timestamp"])
        # 4) Sort within chunk for stable append (final file will still be in order)
        # We do a light sort here to reduce disorder; final guarantee will be okay because subsequent loads sort by timestamp.
        chunk = chunk.sort_values(["customer_ID", "timestamp"])

        # 5) Group by customer within the chunk and append to per-customer files
        for cid, g in chunk.groupby("customer_ID", sort=False):
            write_group(g, OUT)

        n_rows += len(chunk)
        if i % 10 == 0:
            print(f"[chunk {i}] processed rows so far: {n_rows:,}")

    print(f"Done. Wrote per-customer CSVs to: {OUT}")

if __name__ == "__main__":
    main()
