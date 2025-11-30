# src/models/encoder.py
import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    """
    Encodes a sequence [B, T, F] into a single vector [B, d_model]
    using a GRU. Input size = number of features (F).
    """
    def __init__(self, input_dim: int, d_model: int, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        _, h = self.gru(x)           # h: [num_layers, B, d_model]
        return h[-1]                  # last layer: [B, d_model]

def build_encoder(kind: str, input_dim: int, cfg_model: dict) -> nn.Module:
    """
    Factory used by train.py
    kind: "gru", "transformer", "transformer-simple" (supported)
    cfg_model should contain keys: d_model, n_layers, dropout
    """
    kind = (kind or "gru").lower()
    d_model = int(cfg_model.get("d_model", 128))
    n_layers = int(cfg_model.get("n_layers", 1))
    dropout = float(cfg_model.get("dropout", 0.0))

    if kind == "gru":
        return GRUEncoder(input_dim=input_dim, d_model=d_model, n_layers=n_layers, dropout=dropout)
    
    elif kind == "transformer":
        from src.models.transformer_encoder import TransformerEncoder
        n_heads = int(cfg_model.get("n_heads", 4))
        dim_feedforward = int(cfg_model.get("dim_feedforward", d_model * 4))
        return TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    elif kind == "transformer-simple":
        from src.models.transformer_encoder import TemporalTransformerEncoder
        n_heads = int(cfg_model.get("n_heads", 4))
        return TemporalTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
    
    raise ValueError(f"Unknown encoder kind: {kind}. Supported: gru, transformer, transformer-simple")
