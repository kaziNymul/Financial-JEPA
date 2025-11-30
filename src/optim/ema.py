import torch
@torch.no_grad()
def ema_update(target, online, decay=0.996):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_((decay)).add_(p_o.data, alpha=1.0-decay)
