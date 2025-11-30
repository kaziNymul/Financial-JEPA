# src/models/transformer_encoder.py
"""
Temporal Transformer Encoder for JEPA.
Alternative to GRU that better captures long-range dependencies.
Inspired by Vision Transformer architecture from I-JEPA.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for temporal sequences.
    More powerful than GRU for capturing long-range dependencies.
    
    Architecture similar to ViT used in I-JEPA, adapted for time-series.
    """
    def __init__(
        self, 
        input_dim: int, 
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_seq_len: int = 512
    ):
        """
        Args:
            input_dim: Number of input features (F)
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project input features to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # CLS token for sequence representation (like ViT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] input sequence
        Returns:
            [B, d_model] sequence representation
        """
        B, T, F = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # [B, T, d_model]
        
        # Add CLS token (like ViT/BERT)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x)  # [B, T+1, d_model]
        
        # Use CLS token as sequence representation
        cls_output = x[:, 0, :]  # [B, d_model]
        
        return self.norm(cls_output)


class TemporalTransformerEncoder(nn.Module):
    """
    Simplified Transformer encoder that averages over time.
    Lighter alternative to full TransformerEncoder.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F]
        Returns:
            [B, d_model] - mean pooled representation
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Mean pooling over time
        x = x.mean(dim=1)  # [B, d_model]
        return self.norm(x)
