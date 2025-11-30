import os, glob, torch, pandas as pd, torch.nn.functional as F
from src.utils.io import load_ckpt
from src.models.encoder import build_encoder
from src.models.predictor import MLPredictor

def export_anomaly_curve(ckpt_path, processed_glob, features, window, horizon, out_csv):
    ck = load_ckpt(ckpt_path, map_location="cpu"); cfg = ck["cfg"]
    enc = build_encoder(cfg["model"]["encoder"], len(features), cfg["model"]); enc.load_state_dict(ck["enc"]); enc.eval()
    pred = MLPredictor(cfg["model"]["d_model"], cfg["model"]["predictor_hidden"], cfg["model"]["predictor_layers"])
    pred.load_state_dict(ck["pred"]); pred.eval()
    tgt = build_encoder(cfg["model"]["encoder"], len(features), cfg["model"]); tgt.load_state_dict(ck["tgt"]); tgt.eval()

    rows=[]
    for f in sorted(glob.glob(processed_glob)):
        df = pd.read_csv(f).sort_values(cfg["data"]["time_col"])
        X = df[features].values
        for t in range(window, len(X)-horizon+1):
            past = torch.tensor(X[t-window:t], dtype=torch.float32).unsqueeze(0)
            future = torch.tensor(X[t:t+horizon], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                zc = enc(past); zp=pred(zc); zt=tgt(future)
                zp = F.normalize(zp, dim=-1); zt = F.normalize(zt, dim=-1)
                s = (1 - (zp*zt).sum(dim=-1)).item()
            rows.append([df[cfg["data"]["entity_col"]].iloc[t], df[cfg["data"]["time_col"]].iloc[t+horizon-1], s])
    out = pd.DataFrame(rows, columns=["entity_id","timestamp","anomaly_score"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True); out.to_csv(out_csv, index=False)
    print("Wrote", out_csv)
