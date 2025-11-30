import torch.nn as nn
class MLPredictor(nn.Module):
    def __init__(self, d_model=96, hidden=128, layers=2):
        super().__init__()
        blocks=[]; d=d_model
        for _ in range(layers-1):
            blocks += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        blocks += [nn.Linear(d, d_model)]
        self.net = nn.Sequential(*blocks)
    def forward(self, z): return self.net(z)
