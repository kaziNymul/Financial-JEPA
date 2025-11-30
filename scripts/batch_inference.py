import argparse
from src.eval.embeddings import export_embeddings
from src.eval.anomalies import export_anomaly_curve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_cpu/best.pt")
    ap.add_argument("--glob", default="data/processed/bank/*.csv")
    ap.add_argument("--features", default="spend,income,balance,payment_ratio,delinquency_count")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out_dir", default="outputs_cpu")
    a = ap.parse_args()
    feats = a.features.split(",")
    emb_csv = f"{a.out_dir}/embeddings.csv"
    anom_csv = f"{a.out_dir}/anomalies.csv"
    export_embeddings(a.ckpt, a.glob, feats, a.window, emb_csv)
    export_anomaly_curve(a.ckpt, a.glob, feats, a.window, a.horizon, anom_csv)
