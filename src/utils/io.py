from pathlib import Path
import torch, pickle
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def save_ckpt(state, path): ensure_dir(Path(path).parent); torch.save(state, path)
def load_ckpt(path, map_location="cpu"): return torch.load(path, map_location=map_location)
def dump_pkl(obj, path): ensure_dir(Path(path).parent); pickle.dump(obj, open(path,"wb"))
def load_pkl(path): return pickle.load(open(path,"rb"))
