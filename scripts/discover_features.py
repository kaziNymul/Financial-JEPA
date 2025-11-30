# scripts/discover_features.py
import glob, json, numpy as np, polars as pl
from pathlib import Path

DATA_GLOB = "/root/data_amex_shards/*.csv"
EXCLUDE = {"customer_ID","entity_id","timestamp"}
MIN_NUMERIC_RATIO = 0.80         # relax to discover more than 11
PROBE_FILES = 100

def robust_auto_features(files, exclude=EXCLUDE, min_numeric_ratio=MIN_NUMERIC_RATIO, probe_files=PROBE_FILES):
    files = files[: min(probe_files, len(files))]
    cand_counts, cand_numeric = {}, {}
    for p in files:
        try:
            df = pl.read_csv(p, try_parse_dates=True, ignore_errors=True,
                             infer_schema_length=100_000, low_memory=True)
        except Exception:
            continue
        cols = [c for c in df.columns if c not in exclude]
        n = df.height
        if n == 0: continue
        for c in cols:
            cand_counts[c]  = cand_counts.get(c, 0) + n
            arr = df.select(pl.col(c).cast(pl.Float64, strict=False)).to_numpy().ravel()
            cand_numeric[c] = cand_numeric.get(c, 0) + int(np.isfinite(arr).sum())

    keep = []
    for c, cnt in cand_counts.items():
        ratio = cand_numeric.get(c, 0) / max(1, cnt)
        if ratio >= min_numeric_ratio:
            keep.append(c)
    keep.sort()
    return keep

def main():
    files = sorted(glob.glob(DATA_GLOB))
    assert files, f"No files at {DATA_GLOB}"
    feats = robust_auto_features(files)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/candidates.json").write_text(json.dumps(feats, indent=2))
    print(f"Discovered {len(feats)} candidate features → artifacts/candidates.json")

if __name__ == "__main__":
    main()
