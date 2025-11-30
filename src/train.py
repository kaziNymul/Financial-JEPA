# src/train.py
import os, yaml, torch, argparse
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_ckpt
from src.utils.logging import TBLogger
from src.utils.monitoring import RepresentationMonitor
from src.data.streaming_dataset import build_streaming_loaders
from src.data.augmentations import AugmentationPipeline
from src.models.encoder import build_encoder
from src.models.predictor import MLPredictor
from src.optim.ema import ema_update
from src.optim.sched import cosine_warmup
from src.loss import make_loss

def train(cfg_path, sagemaker_args=None):
    # ------------------
    # Load config & seed
    # ------------------
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
            cfg["data"]["processed_glob"] = os.path.join(sagemaker_args.data_dir, "*.csv")

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------
    # Data / loaders (STREAMING - no upfront indexing)
    # ------------------
    feats_cfg = cfg["data"].get("features", ["*"])
    samples_per_file = int(os.environ.get("SAMPLES_PER_FILE", "64"))

    tr, va, scaler = build_streaming_loaders(
        cfg["data"]["processed_glob"],
        feats_cfg,
        cfg["data"].get("window", 6),
        cfg["data"].get("horizon", 1),
        cfg["data"].get("train_ratio", 0.9),
        cfg["train"].get("batch_size", 32),
        samples_per_file=samples_per_file
    )

    # Critical fix: derive true input dim from fitted scaler
    # (works even when cfg uses features: ["*"])
    in_dim = int(len(getattr(scaler, "mean", [])))  # number of feature columns

    # ------------------
    # Model construction
    # ------------------
    mcfg = cfg.get("model", {})
    enc  = build_encoder(mcfg.get("encoder", "gru"), in_dim, mcfg).to(device)
    pred = MLPredictor(
        mcfg.get("d_model", 128),
        mcfg.get("predictor_hidden", 256),
        mcfg.get("predictor_layers", 2)
    ).to(device)

    # EMA teacher (target) — copy of encoder
    tgt  = build_encoder(mcfg.get("encoder", "gru"), in_dim, mcfg).to(device)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad = False

    # ------------------
    # Optim / loss / logging / dirs
    # ------------------
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=cfg["train"].get("lr", 7.5e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-5)
    )
    loss_fn = make_loss(cfg.get("loss", {}).get("type", "cosine"))

    tb = TBLogger(cfg.get("logging", {}).get("tb_dir", "runs"))

    ensure_dir(cfg["train"].get("ckpt_dir", "checkpoints_cpu"))
    ensure_dir(cfg.get("eval", {}).get("out_dir", "outputs_cpu"))

    ckpt_dir = cfg["train"].get("ckpt_dir", "checkpoints_cpu")
    ckpt_every = int(cfg["train"].get("ckpt_every_steps", 0))  # 0 = only save best/last

    # ------------------
    # Masking defaults (JEPA-style block masking)
    # ------------------
    masking = mcfg.get("masking", {})
    time_mask_prob    = float(masking.get("time_mask_prob", 0.0))
    time_mask_span    = int(masking.get("time_mask_span", 2))
    feature_mask_prob = float(masking.get("feature_mask_prob", 0.0))
    feature_mask_span = int(masking.get("feature_mask_span", 2))
    num_time_blocks   = int(masking.get("num_time_blocks", 2))
    num_feature_blocks = int(masking.get("num_feature_blocks", 1))

    # ------------------
    # Data Augmentation (optional)
    # ------------------
    aug_config = cfg.get("augmentation", {})
    augmenter = AugmentationPipeline(aug_config) if aug_config else None

    # ------------------
    # Representation Quality Monitoring
    # ------------------
    monitor = RepresentationMonitor(log_every=cfg["train"].get("log_every_steps", 200))

    # ------------------
    # Training loop
    # ------------------
    max_epochs = int(cfg["train"].get("max_epochs", 6))
    early_stop_pat = int(cfg["train"].get("early_stop_patience", 3))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    log_every = int(cfg["train"].get("log_every_steps", 200))

    # warmup/total steps for cosine schedule
    # For streaming datasets, use max_steps_per_epoch env var instead of len()
    max_steps_per_epoch = int(os.environ.get("MAX_STEPS_PER_EPOCH", "1000"))
    try:
        steps_per_epoch = max(1, len(tr))
    except TypeError:
        # IterableDataset has no len(), use max_steps_per_epoch
        steps_per_epoch = max_steps_per_epoch
    max_steps = max_epochs * steps_per_epoch
    warmup = int(cfg["train"].get("warmup_steps", 200))

    step = 0
    best_val = float("inf")
    patience = 0

    for epoch in range(max_epochs):
        enc.train(); pred.train()
        epoch_steps = 0
        
        for past, future in tr:
            if epoch_steps >= max_steps_per_epoch:
                print(f"[epoch {epoch+1}] Reached max steps ({max_steps_per_epoch}), moving to validation")
                break
            
            past, future = past.to(device), future.to(device)
            epoch_steps += 1

            # --- Data augmentation (if enabled)
            if augmenter is not None:
                past = augmenter(past)

            # --- JEPA-style block masking for temporal sequences
            if time_mask_prob > 0.0 and time_mask_span > 0 and num_time_blocks > 0:
                B, T, F = past.shape
                # Block masking: mask contiguous spans like I-JEPA
                for b in range(B):
                    for _ in range(num_time_blocks):
                        if T > time_mask_span:
                            start = torch.randint(0, T - time_mask_span + 1, (1,)).item()
                            past[b, start:start + time_mask_span, :] = 0.0

            # --- JEPA-style block masking for features
            if feature_mask_prob > 0.0 and feature_mask_span > 0 and num_feature_blocks > 0:
                B, T, F = past.shape
                # Block masking: mask contiguous feature spans
                for b in range(B):
                    for _ in range(num_feature_blocks):
                        if F > feature_mask_span:
                            start = torch.randint(0, F - feature_mask_span + 1, (1,)).item()
                            past[b, :, start:start + feature_mask_span] = 0.0

            # --- forward / loss
            Zc = enc(past)               # [B, d_model] context encoder
            Zp = pred(Zc)                # [B, d_model] predictor output
            Zt = tgt(future)             # [B, d_model] target encoder (stop-grad via EMA)
            loss = loss_fn(Zp, Zt)

            # --- Monitor representation quality
            if step % log_every == 0:
                repr_metrics = monitor.update(Zc, Zp, Zt, tb)

            # --- LR schedule
            lr = cosine_warmup(step, warmup=warmup, max_steps=max_steps, base_lr=cfg["train"].get("lr", 7.5e-4))
            for g in opt.param_groups:
                g["lr"] = lr

            # --- step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            ema_update(tgt, enc, mcfg.get("ema_decay", 0.997))

            # --- logs / ckpt
            if step % log_every == 0:
                tb.log_scalar("train/loss", float(loss.item()), step)
                tb.log_scalar("train/lr", lr, step)

            if ckpt_every > 0 and step > 0 and (step % ckpt_every == 0):
                save_ckpt(
                    {"enc": enc.state_dict(), "pred": pred.state_dict(), "tgt": tgt.state_dict(), "cfg": cfg},
                    os.path.join(ckpt_dir, f"step_{step}.pt")
                )
            step += 1

        # ------------------
        # Epoch-end validation
        # ------------------
        val_loss = evaluate(enc, pred, tgt, va, loss_fn, device)
        tb.log_scalar("val/loss", val_loss, step)
        print(f"[epoch {epoch}] val loss {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            save_ckpt(
                {"enc": enc.state_dict(), "pred": pred.state_dict(), "tgt": tgt.state_dict(), "cfg": cfg},
                os.path.join(ckpt_dir, "best.pt")
            )
        else:
            patience += 1
            if early_stop_pat > 0 and patience >= early_stop_pat:
                print("Early stopping.")
                break

    tb.close()

def evaluate(enc, pred, tgt, loader, loss_fn, device):
    enc.eval(); pred.eval(); tgt.eval()
    tot = 0.0; n = 0
    with torch.no_grad():
        for past, future in loader:
            past, future = past.to(device), future.to(device)
            l = loss_fn(pred(enc(past)), tgt(future)).item()
            tot += l * past.size(0)
            n += past.size(0)
    return tot / max(1, n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/sagemaker.yaml")
    
    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "checkpoints"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "data/processed/data_amex_shards"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "outputs"))
    
    # Hyperparameters (tuned by SageMaker)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--time-mask-prob", type=float, default=0.15)
    parser.add_argument("--feature-mask-prob", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--mixed-precision", type=int, default=1)
    parser.add_argument("--encoder", type=str, default="gru")
    parser.add_argument("--all-features", type=int, default=1)
    parser.add_argument("--sagemaker-mode", type=int, default=0)
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 STARTING FINANCIAL-JEPA TRAINING")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Hyperparameters:")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model Dim: {args.d_model}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  EMA Decay: {args.ema_decay}")
    print(f"  Time Mask: {args.time_mask_prob}")
    print(f"  Feature Mask: {args.feature_mask_prob}")
    print("="*80 + "\n")
    
    train(args.config, args)
