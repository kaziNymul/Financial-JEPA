#!/usr/bin/env python3
import glob
import json
import polars as pl
from pathlib import Path

SHARDS = "/mnt/e/JEPA/financial-jepa/financial-jepa/data/processed/data_amex_shards/*.csv"   # <-- update if shard path is different
OUT = Path("artifacts/features.json")

def main():
    files = sorted(glob.glob(SHARDS))
    if not files:
        raise SystemExit(f"No shard files found at {SHARDS}")

    print(f"[feature scan] probing {files[0]} ...")

    # Read first shard fully to inspect schema
    df = pl.read_csv(
        files[0],
        try_parse_dates=True,
        ignore_errors=True,
        infer_schema_length=20000,
        low_memory=True,
    )

    # Identify numeric-like columns
    numeric_feats = []
    for col, dtype in zip(df.columns, df.dtypes):
        dtype_str = str(dtype)
        if dtype_str.startswith("Float") or dtype_str.startswith("Int"):
            if col not in ("customer_ID", "entity_id", "S_2", "timestamp"):
                numeric_feats.append(col)

    numeric_feats = sorted(numeric_feats)
    print(f"[feature scan] found {len(numeric_feats)} numeric-like features")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(numeric_feats, indent=2))
    print(f"[feature scan] saved features → {OUT}")

if __name__ == "__main__":
    main()
