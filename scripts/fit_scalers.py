# scripts/fit_scalers.py
import sys, glob, json, numpy as np
from pathlib import Path

# allow "from src..." imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.dataset import _read_entity_pl, RunningStandardizer

SHARDS = "/mnt/e/JEPA/financial-jepa/financial-jepa/data/processed/data_amex_shards/*.csv"
FEATS  = Path("artifacts/features.json")
OUT    = Path("artifacts/scaler.npz")

def main():
    feats = json.loads(FEATS.read_text())
    files = sorted(glob.glob(SHARDS))
    print(f"[fit_scaler] found {len(files)} shard files in pattern: {SHARDS}")
    if not files:
        raise SystemExit("No shard files found — fix SHARDS path in fit_scalers.py")

    sc = RunningStandardizer(n_features=len(feats))
    cnt = 0
    for f in files:
        X = _read_entity_pl(f, feats)
        if X is None or len(X) == 0:
            # debug: tell which file had no valid rows/columns
            print(f"[fit_scaler] skip (no valid rows for selected features): {f}")
            continue
        sc.partial_fit(X)
        cnt += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, mean=sc.mean, std=sc.scale_)
    print(f"Scaler fit on {cnt} files → {OUT}")

if __name__ == "__main__":
    main()
