from torch.utils.tensorboard import SummaryWriter
class TBLogger:
    def __init__(self, logdir): self.w = SummaryWriter(logdir)
    def log_scalar(self, k, v, step): self.w.add_scalar(k, v, step)
    def close(self): self.w.close()
