import argparse
from src.train import train
from src.eval.embeddings import export_embeddings
from src.eval.anomalies import export_anomaly_curve
from src.eval.visualize import plot_umap

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    tr = sub.add_parser("train"); tr.add_argument("--config", default="configs/small.yaml")
    em = sub.add_parser("emb"); em.add_argument("--ckpt", required=True); em.add_argument("--glob", required=True)
    em.add_argument("--features", required=True); em.add_argument("--window", type=int, default=8); em.add_argument("--out", default="outputs_cpu/embeddings.csv")
    an = sub.add_parser("anomaly"); an.add_argument("--ckpt", required=True); an.add_argument("--glob", required=True)
    an.add_argument("--features", required=True); an.add_argument("--window", type=int, default=8); an.add_argument("--horizon", type=int, default=1); an.add_argument("--out", default="outputs_cpu/anomalies.csv")
    vz = sub.add_parser("umap"); vz.add_argument("--emb", required=True); vz.add_argument("--out", default="outputs_cpu/umap.png")
    vz.add_argument("--neighbors", type=int, default=15); vz.add_argument("--min_dist", type=float, default=0.3)

    args = ap.parse_args()
    if args.cmd=="train": train(args.config)
    elif args.cmd=="emb": export_embeddings(args.ckpt, args.glob, args.features.split(","), args.window, args.out)
    elif args.cmd=="anomaly": export_anomaly_curve(args.ckpt, args.glob, args.features.split(","), args.window, args.horizon, args.out)
    elif args.cmd=="umap": plot_umap(args.emb, out_png=args.out, n_neighbors=args.neighbors, min_dist=args.min_dist)
    else: ap.print_help()
