import glob, time, argparse, requests, pandas as pd

def iter_rows(glob_path):
    files = sorted(glob.glob(glob_path))
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        for _, r in df.iterrows():
            yield {
                "entity_id": str(r["entity_id"]),
                "timestamp": r["timestamp"].to_pydatetime().isoformat(),
                "features": {k: float(r[k]) for k in df.columns if k not in ["entity_id","timestamp"]}
            }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help='e.g. "data/processed/amex/*.csv"')
    ap.add_argument("--ingest_url", default="http://localhost:8001/ingest/event")
    ap.add_argument("--rate_hz", type=float, default=10.0, help="events per second")
    args = ap.parse_args()

    delay = 1.0 / max(0.1, args.rate_hz)
    s = requests.Session()
    sent = 0
    for row in iter_rows(args.glob):
        resp = s.post(args.ingest_url, json=row, timeout=10)
        if not resp.ok:
            print("POST failed:", resp.status_code, resp.text)
            break
        sent += 1
        if sent % 100 == 0:
            print("sent", sent, "events")
        time.sleep(delay)
    print("done, sent", sent, "events")
