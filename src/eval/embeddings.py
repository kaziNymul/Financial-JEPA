import os, glob, torch, pandas as pd
from src.utils.io import load_ckpt
from src.models.encoder import build_encoder

def export_embeddings(ckpt_path, processed_glob, features, window, out_csv):
    ck = load_ckpt(ckpt_path, map_location="cpu"); cfg = ck["cfg"]
    enc = build_encoder(cfg["model"]["encoder"], len(features), cfg["model"])
    enc.load_state_dict(ck["enc"]); enc.eval()
    rows=[]
    for f in sorted(glob.glob(processed_glob)):
        df = pd.read_csv(f).sort_values(cfg["data"]["time_col"])
        X = df[features].values
        for t in range(window, len(X)):
            past = torch.tensor(X[t-window:t], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad(): z = enc(past).squeeze(0).numpy()
            rows.append([df[cfg["data"]["entity_col"]].iloc[t], df[cfg["data"]["time_col"]].iloc[t]] + z.tolist())
    if not rows:
        raise RuntimeError("No rows produced. Check data/window.")
    cols = ["entity_id","timestamp"] + [f"z{i}" for i in range(len(rows[0])-2)]
    out = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True); out.to_csv(out_csv, index=False)
    print("Wrote", out_csv)
