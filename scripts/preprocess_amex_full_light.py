# scripts/preprocess_amex_full_light.py
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/amex/train_data.csv")
    ap.add_argument("--out", default="data/processed/amex")
    ap.add_argument("--chunk", type=int, default=50_000)  # SMALLER
    ap.add_argument("--time-col", default="S_2")
    return ap.parse_args()

def append_group(g, out_dir: Path):
    cid = g["customer_ID"].iloc[0]
    f = out_dir / f"{cid}.csv"
    mode = "a" if f.exists() else "w"
    header = not f.exists()
    g.to_csv(f, index=False, mode=mode, header=header)

def main():
    a = parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # Read header to get all columns
    head = pd.read_csv(a.raw, nrows=5, low_memory=False)
    cols = head.columns.tolist()
    if "customer_ID" not in cols: raise SystemExit("customer_ID missing")
    if a.time_col not in cols:    raise SystemExit(f"{a.time_col} missing")

    # Stream small chunks; NO sorting inside chunk
    reader = pd.read_csv(
        a.raw,
        usecols=cols,               # keep ALL columns
        dtype={"customer_ID": "string", a.time_col: "string"},
        chunksize=a.chunk,
        low_memory=False,
        memory_map=True,
    )

    n=0
    for i, chunk in enumerate(reader, 1):
        chunk["timestamp"] = pd.to_datetime(chunk[a.time_col], errors="coerce")
        chunk = chunk.dropna(subset=["customer_ID","timestamp"])
        # group within the chunk and append out
        for cid, g in chunk.groupby("customer_ID", sort=False):
            append_group(g, out)
        n += len(chunk)
        if i % 20 == 0:
            print(f"[chunk {i}] rows processed: {n:,}")

    print(f"Done. Wrote per-customer CSVs to {out}")

if __name__ == "__main__":
    main()
