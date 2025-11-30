# src/utils/monitoring.py
"""
Utilities for monitoring JEPA training quality.
Helps detect representation collapse and track embedding quality.
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_representation_metrics(embeddings):
    """
    Compute metrics to monitor representation quality.
    
    Args:
        embeddings: [B, d_model] tensor of representations
    
    Returns:
        dict with metrics:
            - std_mean: average std across dimensions (collapse indicator)
            - std_std: variance of stds (uniformity)
            - norm_mean: average L2 norm
            - rank: effective rank of representation matrix
    """
    embeddings = embeddings.detach().cpu()
    B, D = embeddings.shape
    
    # Standard deviation per dimension (should not collapse to 0)
    stds = embeddings.std(dim=0)
    std_mean = stds.mean().item()
    std_std = stds.std().item()
    
    # L2 norm (should be relatively stable)
    norms = torch.norm(embeddings, dim=1)
    norm_mean = norms.mean().item()
    
    # Effective rank (measures diversity of representations)
    # Based on singular values
    try:
        _, S, _ = torch.svd(embeddings)
        S_normalized = S / S.sum()
        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()
    except:
        effective_rank = -1.0
    
    return {
        'std_mean': std_mean,
        'std_std': std_std,
        'norm_mean': norm_mean,
        'effective_rank': effective_rank,
    }


def compute_alignment_uniformity(z_context, z_target, t=2.0):
    """
    Compute alignment and uniformity metrics for self-supervised learning.
    
    From "Understanding Contrastive Representation Learning through Alignment 
    and Uniformity on the Hypersphere" (Wang & Isola, 2020)
    
    Args:
        z_context: [B, d_model] context representations
        z_target: [B, d_model] target representations
        t: temperature for uniformity
    
    Returns:
        dict with:
            - alignment: how well positive pairs align (lower is better)
            - uniformity: how uniform representations are (lower is better)
    """
    z_context = F.normalize(z_context.detach(), dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    
    # Alignment: average distance between positive pairs
    alignment = (z_context - z_target).pow(2).sum(dim=1).mean()
    
    # Uniformity: average pairwise Gaussian potential
    B = z_context.shape[0]
    if B > 1:
        # Compute pairwise distances
        z_all = torch.cat([z_context, z_target], dim=0)
        sq_dists = torch.cdist(z_all, z_all).pow(2)
        uniformity = torch.exp(-t * sq_dists).mean().log()
    else:
        uniformity = torch.tensor(0.0)
    
    return {
        'alignment': alignment.item(),
        'uniformity': uniformity.item(),
    }


def check_representation_collapse(embeddings, threshold=0.01):
    """
    Check if representations have collapsed.
    
    Args:
        embeddings: [B, d_model] tensor
        threshold: minimum acceptable std per dimension
    
    Returns:
        bool: True if collapse detected
        float: fraction of collapsed dimensions
    """
    stds = embeddings.std(dim=0)
    collapsed = (stds < threshold).float().mean()
    return collapsed.item() > 0.5, collapsed.item()


def compute_predictor_variance_ratio(z_pred, z_target):
    """
    Compute ratio of predictor output variance to target variance.
    Should be close to 1.0 for good learning.
    
    Args:
        z_pred: [B, d_model] predicted representations
        z_target: [B, d_model] target representations
    
    Returns:
        float: variance ratio
    """
    var_pred = z_pred.var(dim=0).mean()
    var_target = z_target.var(dim=0).mean()
    
    ratio = (var_pred / (var_target + 1e-8)).item()
    return ratio


class RepresentationMonitor:
    """
    Monitor representation quality during training.
    """
    
    def __init__(self, log_every=100):
        self.log_every = log_every
        self.step = 0
    
    def update(self, z_context, z_pred, z_target, logger=None):
        """
        Compute and optionally log representation metrics.
        
        Args:
            z_context: [B, d_model] context encoder output
            z_pred: [B, d_model] predictor output
            z_target: [B, d_model] target encoder output
            logger: TBLogger instance (optional)
        
        Returns:
            dict with all metrics
        """
        self.step += 1
        
        # Compute metrics
        metrics = {}
        
        # Basic representation metrics
        context_metrics = compute_representation_metrics(z_context)
        metrics.update({f'repr/context_{k}': v for k, v in context_metrics.items()})
        
        pred_metrics = compute_representation_metrics(z_pred)
        metrics.update({f'repr/pred_{k}': v for k, v in pred_metrics.items()})
        
        target_metrics = compute_representation_metrics(z_target)
        metrics.update({f'repr/target_{k}': v for k, v in target_metrics.items()})
        
        # Alignment & uniformity
        align_unif = compute_alignment_uniformity(z_pred, z_target)
        metrics.update({f'repr/{k}': v for k, v in align_unif.items()})
        
        # Collapse detection
        collapsed, collapse_frac = check_representation_collapse(z_pred)
        metrics['repr/collapse_fraction'] = collapse_frac
        metrics['repr/collapsed'] = float(collapsed)
        
        # Variance ratio
        var_ratio = compute_predictor_variance_ratio(z_pred, z_target)
        metrics['repr/variance_ratio'] = var_ratio
        
        # Log to tensorboard
        if logger is not None and self.step % self.log_every == 0:
            for key, value in metrics.items():
                logger.log_scalar(key, value, self.step)
        
        return metrics
