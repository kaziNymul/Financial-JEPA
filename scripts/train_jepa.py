import argparse
from src.train import train
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    args = ap.parse_args()
    train(args.config)
