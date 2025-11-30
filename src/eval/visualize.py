import os, pandas as pd, umap
import matplotlib.pyplot as plt

def plot_umap(emb_csv, color_col=None, out_png="outputs_cpu/umap.png", n_neighbors=15, min_dist=0.3):
    df = pd.read_csv(emb_csv)
    Z = df[[c for c in df.columns if c.startswith("z")]].values
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    XY = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    if color_col and color_col in df.columns:
        plt.scatter(XY[:,0], XY[:,1], s=6, c=df[color_col].values, alpha=0.7)
    else:
        plt.scatter(XY[:,0], XY[:,1], s=6, alpha=0.7)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160)
    print("Saved", out_png)
