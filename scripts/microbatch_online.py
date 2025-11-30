import time, argparse
from pathlib import Path
from src.train import train as full_train
from src.eval.anomalies import export_anomaly_curve

DELTA_DIR = Path("data/stream/delta_entities")

def clear_touches():
    if DELTA_DIR.exists():
        for p in DELTA_DIR.glob("*.touch"):
            p.unlink()

def any_touches():
    return DELTA_DIR.exists() and any(DELTA_DIR.glob("*.touch"))

def run_online_cycle(cfg_online: str, features: list[str], window: int, horizon: int):
    full_train(cfg_online)
    export_anomaly_curve(
        ckpt_path="checkpoints_cpu/best.pt",
        processed_glob="data/processed/bank/*.csv",
        features=features,
        window=window,
        horizon=horizon,
        out_csv="outputs_cpu/anomalies.csv"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/online.yaml")
    ap.add_argument("--interval_sec", type=int, default=60)
    ap.add_argument("--features", default="spend,income,balance,payment_ratio,delinquency_count")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    feats = args.features.split(",")
    print(f"[online] loop start; every {args.interval_sec}s")
    clear_touches()

    while True:
        try:
            if any_touches():
                print("[online] data changed → fine-tuning...")
                run_online_cycle(args.config, feats, args.window, args.horizon)
                clear_touches()
                print("[online] cycle done.")
            else:
                print("[online] no new data; sleeping...")
        except Exception as e:
            print("[online] error:", e)
        time.sleep(args.interval_sec)
