#!/usr/bin/env python3
"""
SIMPLE ROBUST TRAINING SCRIPT FOR SAGEMAKER
- Downloads data once at start
- Standard PyTorch training loop
- No complex streaming
"""
import os, yaml, torch, argparse, time
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_ckpt
from src.data.simple_dataset import build_simple_loaders
from src.data.augmentations import AugmentationPipeline
from src.models.encoder import build_encoder
from src.models.predictor import MLPredictor
from src.optim.ema import ema_update
from src.optim.sched import cosine_warmup
from src.loss import make_loss
import numpy as np


def train(cfg_path, sagemaker_args=None):
    print("=" * 80)
    print("🚀 STARTING SIMPLE FINANCIAL-JEPA TRAINING")
    print("=" * 80)
    
    # Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Override with SageMaker hyperparameters
    if sagemaker_args:
        if hasattr(sagemaker_args, 'learning_rate'):
            cfg["train"]["lr"] = float(sagemaker_args.learning_rate)
        if hasattr(sagemaker_args, 'batch_size'):
            cfg["train"]["batch_size"] = int(sagemaker_args.batch_size)
        if hasattr(sagemaker_args, 'd_model'):
            cfg["model"]["d_model"] = int(sagemaker_args.d_model)
        if hasattr(sagemaker_args, 'n_layers'):
            cfg["model"]["n_layers"] = int(sagemaker_args.n_layers)
        if hasattr(sagemaker_args, 'dropout'):
            cfg["model"]["dropout"] = float(sagemaker_args.dropout)
        if hasattr(sagemaker_args, 'ema_decay'):
            cfg["model"]["ema_decay"] = float(sagemaker_args.ema_decay)
        if hasattr(sagemaker_args, 'weight_decay'):
            cfg["train"]["weight_decay"] = float(sagemaker_args.weight_decay)
        if hasattr(sagemaker_args, 'grad_clip'):
            cfg["train"]["grad_clip"] = float(sagemaker_args.grad_clip)
        if hasattr(sagemaker_args, 'time_mask_prob'):
            cfg["model"]["masking"]["time_mask_prob"] = float(sagemaker_args.time_mask_prob)
        if hasattr(sagemaker_args, 'feature_mask_prob'):
            cfg["model"]["masking"]["feature_mask_prob"] = float(sagemaker_args.feature_mask_prob)
        
        # SageMaker paths
        if hasattr(sagemaker_args, 'model_dir'):
            cfg["train"]["ckpt_dir"] = sagemaker_args.model_dir
        if hasattr(sagemaker_args, 'data_dir'):
            data_dir = sagemaker_args.data_dir
        else:
            data_dir = "/opt/ml/input/data/training"
    else:
        data_dir = cfg["data"].get("processed_dir", "data")
    
    set_seed(cfg.get("seed", 42))
    
    # Print config
    print(f"Config: {cfg_path}")
    print(f"Data Dir: {data_dir}")
    print("Hyperparameters:")
    print(f"  Learning Rate: {cfg['train']['lr']}")
    print(f"  Batch Size: {cfg['train']['batch_size']}")
    print(f"  Model Dim: {cfg['model']['d_model']}")
    print(f"  Layers: {cfg['model']['n_layers']}")
    print(f"  Dropout: {cfg['model']['dropout']}")
    print(f"  EMA Decay: {cfg['model'].get('ema_decay', 0.996)}")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ------------------
    # DATA
    # ------------------
    print("\n📊 LOADING DATA...")
    start_time = time.time()
    
    # Use S3 path if in SageMaker mode
    if hasattr(sagemaker_args, 'sagemaker_mode') and sagemaker_args.sagemaker_mode:
        s3_data_path = "s3://financial-jepa-data-057149785966/processed"
        scaler_path = "s3://financial-jepa-data-057149785966/artifacts/scaler.npz"
    else:
        s3_data_path = data_dir
        scaler_path = "artifacts/scaler.npz"
    
    train_loader, val_loader, n_features = build_simple_loaders(
        data_dir=s3_data_path,
        scaler_path=scaler_path,
        batch_size=cfg["train"]["batch_size"],
        past_len=cfg["data"]["past_len"],
        future_len=cfg["data"]["future_len"],
        num_workers=4
    )
    
    data_time = time.time() - start_time
    print(f"✅ Data loaded in {data_time:.1f}s")
    print(f"Features: {n_features}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ------------------
    # AUGMENTATION
    # ------------------
    aug = AugmentationPipeline(
        time_mask_prob=cfg["model"]["masking"]["time_mask_prob"],
        feature_mask_prob=cfg["model"]["masking"]["feature_mask_prob"],
        noise_std=cfg["model"]["masking"].get("noise_std", 0.0)
    )
    
    # ------------------
    # MODEL
    # ------------------
    print("\n🏗️  BUILDING MODEL...")
    model_cfg = cfg["model"].copy()
    model_cfg["n_features"] = n_features
    
    enc = build_encoder(model_cfg).to(device)
    pred = MLPredictor(model_cfg["d_model"], n_features).to(device)
    
    # EMA encoder
    enc_ema = build_encoder(model_cfg).to(device)
    enc_ema.load_state_dict(enc.state_dict())
    for p in enc_ema.parameters():
        p.requires_grad = False
    
    # Count parameters
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in pred.parameters())
    print(f"✅ Model built: {n_params:,} parameters")
    
    # ------------------
    # OPTIMIZER
    # ------------------
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["train"].get("weight_decay", 1e-5)
    )
    
    loss_fn = make_loss()
    
    # ------------------
    # TRAINING SETTINGS
    # ------------------
    max_epochs = int(cfg["train"].get("max_epochs", 12))
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 3))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    log_every = int(cfg["train"].get("log_every_steps", 100))
    ema_decay = float(cfg["model"].get("ema_decay", 0.996))
    
    steps_per_epoch = len(train_loader)
    max_steps = max_epochs * steps_per_epoch
    warmup_steps = int(cfg["train"].get("warmup_steps", 200))
    
    print(f"\n📈 TRAINING CONFIG:")
    print(f"  Epochs: {max_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {max_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Grad clip: {grad_clip}")
    print(f"  Early stop patience: {early_stop_patience}")
    
    # ------------------
    # TRAINING LOOP
    # ------------------
    print("\n" + "=" * 80)
    print("🎯 STARTING TRAINING")
    print("=" * 80)
    
    step = 0
    best_val_loss = float("inf")
    patience = 0
    
    for epoch in range(max_epochs):
        enc.train()
        pred.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        epoch_start = time.time()
        
        for batch_idx, (past, future) in enumerate(train_loader):
            past = past.to(device)
            future = future.to(device)
            
            # Augment
            past_aug, mask_t, mask_f = aug(past)
            
            # Forward
            z_past = enc(past_aug)
            z_future_target = enc_ema(future).detach()
            pred_future = pred(z_past)
            
            # Loss
            loss = loss_fn(pred_future, z_future_target, mask_t, mask_f)
            
            # Backward
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            
            # Learning rate schedule
            lr = cosine_warmup(step, warmup_steps, max_steps, cfg["train"]["lr"], 1e-6)
            for pg in opt.param_groups:
                pg["lr"] = lr
            
            opt.step()
            
            # EMA update
            ema_update(enc, enc_ema, ema_decay)
            
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1
            
            # Log
            if step % log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                print(f"Epoch {epoch+1}/{max_epochs} | Step {step}/{max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        train_loss = epoch_loss / epoch_steps
        print(f"\n✅ Epoch {epoch+1}/{max_epochs} complete in {epoch_time:.1f}s")
        print(f"   Train Loss: {train_loss:.4f}")
        
        # ------------------
        # VALIDATION
        # ------------------
        enc.eval()
        pred.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for past, future in val_loader:
                past = past.to(device)
                future = future.to(device)
                
                z_past = enc(past)
                z_future_target = enc_ema(future)
                pred_future = pred(z_past)
                
                loss = loss_fn(pred_future, z_future_target)
                val_loss += loss.item()
                val_steps += 1
        
        val_loss = val_loss / val_steps
        print(f"   Val Loss: {val_loss:.4f}")
        
        # Report to SageMaker
        if hasattr(sagemaker_args, 'sagemaker_mode') and sagemaker_args.sagemaker_mode:
            # SageMaker looks for this exact format
            print(f"#quality_metric: name=validation:loss, iteration={epoch+1}, value={val_loss}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            
            ckpt_dir = cfg["train"].get("ckpt_dir", "checkpoints")
            ensure_dir(ckpt_dir)
            
            save_ckpt(
                os.path.join(ckpt_dir, "best.pt"),
                enc, enc_ema, pred, opt, epoch, step, val_loss
            )
            print(f"   💾 Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience += 1
            print(f"   Patience: {patience}/{early_stop_patience}")
        
        # Early stopping
        if patience >= early_stop_patience:
            print(f"\n⚠️  Early stopping triggered (no improvement for {early_stop_patience} epochs)")
            break
        
        print()
    
    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/jepa_improved.yaml")
    
    # SageMaker hyperparameters
    parser.add_argument("--sagemaker-mode", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--time-mask-prob", type=float, default=None)
    parser.add_argument("--feature-mask-prob", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    
    # SageMaker paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "checkpoints"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "data"))
    
    args = parser.parse_args()
    
    train(args.config, args)
