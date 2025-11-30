# scripts/screen_features.py
import glob, json, numpy as np, polars as pl
from pathlib import Path

DATA_GLOB = "/root/data_amex_shards/*.csv"
CAND_PATH = Path("artifacts/candidates.json")
OUT_PATH  = Path("artifacts/screened.json")

# thresholds
MAX_MISSING_RATIO = 0.30   # drop columns with >30% NaNs after casting
MIN_STD = 1e-6             # drop (almost) constant columns

def main():
    feats = json.loads(CAND_PATH.read_text())
    files = sorted(glob.glob(DATA_GLOB))[:50]  # small probe to estimate stats
    assert files, f"No files at {DATA_GLOB}"

    miss_cnt = {c: 0 for c in feats}
    n_tot    = {c: 0 for c in feats}
    sums     = {c: 0.0 for c in feats}
    sums2    = {c: 0.0 for c in feats}

    for p in files:
        try:
            df = pl.read_csv(p, try_parse_dates=True, ignore_errors=True,
                             infer_schema_length=100_000, low_memory=True)
        except Exception:
            continue
        have = [c for c in feats if c in df.columns]
        if not have: continue
        df = df.select([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in have])
        arr = df.to_numpy()  # shape [rows, len(have)]
        mask = np.isfinite(arr)
        for j, c in enumerate(have):
            col = arr[:, j]
            n = len(col)
            n_tot[c] += n
            miss_cnt[c] += int((~mask[:, j]).sum())
            valid = col[mask[:, j]]
            if valid.size > 0:
                sums[c]  += float(valid.sum())
                sums2[c] += float((valid**2).sum())

    keep = []
    for c in feats:
        if n_tot[c] == 0: continue
        miss_ratio = miss_cnt[c] / n_tot[c]
        mean = sums[c] / max(1, (n_tot[c] - miss_cnt[c]))
        var  = (sums2[c] / max(1, (n_tot[c]-miss_cnt[c]))) - mean*mean
        std  = float(np.sqrt(max(0.0, var)))
        if miss_ratio <= MAX_MISSING_RATIO and std >= MIN_STD:
            keep.append(c)

    OUT_PATH.write_text(json.dumps(sorted(keep), indent=2))
    print(f"Screened → kept {len(keep)} / {len(feats)} features → {OUT_PATH}")

if __name__ == "__main__":
    main()
