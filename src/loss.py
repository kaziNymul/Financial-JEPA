import torch.nn.functional as F
def cosine_loss(pred, tgt):
    pred = F.normalize(pred, dim=-1); tgt = F.normalize(tgt, dim=-1)
    return 1 - (pred * tgt).sum(dim=-1).mean()
def l2_loss(pred, tgt): return ((pred - tgt)**2).mean()
def make_loss(name): return cosine_loss if name=="cosine" else l2_loss
