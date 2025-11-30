import argparse, subprocess
from pathlib import Path

def run_cmd(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

def dl_amex():
    out = Path("data/raw/amex"); out.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "competitions", "download", "-c", "amex-default-prediction", "-p", str(out)])
    run_cmd(["bash", "-lc", f"cd {out} && unzip -o *.zip"])

def dl_jpx():
    out = Path("data/raw/jpx"); out.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "competitions", "download", "-c", "jpx-tokyo-stock-exchange-prediction", "-p", str(out)])
    run_cmd(["bash", "-lc", f"cd {out} && unzip -o *.zip"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["amex","jpx"], required=True)
    args = ap.parse_args()
    if args.dataset=="amex": dl_amex()
    else: dl_jpx()
