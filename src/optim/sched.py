import math
def cosine_warmup(step, warmup, max_steps, base_lr):
    if max_steps <= 0: return base_lr
    if step < warmup:
        return base_lr * step / max(warmup,1)
    t = (step-warmup)/max(1, max_steps-warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))
