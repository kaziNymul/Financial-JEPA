#!/usr/bin/env python3
"""
COMPLETE AWS TRAINING PIPELINE
1. Download AMEX data from Kaggle (in AWS)
2. Preprocess and chunk data
3. Train JEPA model
4. Save to S3

Everything happens in SageMaker - no local preprocessing needed!
"""
import os
import sys
import yaml
import torch
import argparse
import time
import subprocess
import numpy as np
import polars as pl
from pathlib import Path
from collections import defaultdict

# Setup Kaggle
def setup_kaggle():
    """Setup Kaggle credentials in SageMaker"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy kaggle.json if it exists in the code directory
    if os.path.exists('/opt/ml/code/kaggle.json'):
        import shutil
        shutil.copy('/opt/ml/code/kaggle.json', kaggle_dir / 'kaggle.json')
        os.chmod(kaggle_dir / 'kaggle.json', 0o600)
        print("✅ Kaggle credentials configured")
        return True
    else:
        print("⚠️  No kaggle.json found - will skip Kaggle download")
        return False

def download_amex_data(data_dir="/tmp/amex_raw"):
    """Download AMEX data from Kaggle"""
    os.makedirs(data_dir, exist_ok=True)
    
    if setup_kaggle():
        print("\n📥 Downloading AMEX data from Kaggle...")
        try:
            zip_path = f'{data_dir}/amex-default-prediction.zip'
            subprocess.run([
                'kaggle', 'competitions', 'download', '-c', 
                'amex-default-prediction', '-p', data_dir
            ], check=True)
            
            # Unzip
            print("📦 Extracting data...")
            subprocess.run(['unzip', '-o', zip_path, '-d', data_dir], 
                         check=True, stdout=subprocess.DEVNULL)
            
            # Delete zip to save disk space
            print("🗑️  Removing zip file to save space...")
            os.remove(zip_path)
            
            print("✅ AMEX data downloaded")
            return data_dir
        except Exception as e:
            print(f"❌ Kaggle download failed: {e}")
            print("Will check for existing data...")
    
    # Check if data already exists (from previous run or mounted volume)
    if os.path.exists(os.path.join(data_dir, 'train_data.csv')):
        print(f"✅ Using existing data in {data_dir}")
        return data_dir
    
    # Check SageMaker input channel
    sm_data_dir = '/opt/ml/input/data/training'
    if os.path.exists(sm_data_dir) and os.listdir(sm_data_dir):
        print(f"✅ Using data from SageMaker input: {sm_data_dir}")
        return sm_data_dir
    
    raise FileNotFoundError("No AMEX data found! Please provide data via Kaggle or S3")

def preprocess_amex_data(raw_dir, output_dir="/tmp/processed", chunk_size=50000):
    """
    Preprocess AMEX data:
    - Group by customer_ID
    - Create time series sequences
    - Compute normalization statistics
    - Save in chunks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🔧 PREPROCESSING AMEX DATA")
    print("=" * 80)
    
    train_file = os.path.join(raw_dir, 'train_data.csv')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"train_data.csv not found in {raw_dir}")
    
    print(f"Reading: {train_file}")
    
    # PASS 1: Read header and compute statistics in chunks (memory efficient)
    print("📊 PASS 1: Computing normalization statistics (streaming)...")
    start = time.time()
    
    # Read header
    header_df = pl.read_csv(train_file, n_rows=1)
    numeric_cols = [c for c in header_df.columns if c not in ['customer_ID', 'S_2']]
    print(f"   Numeric features: {len(numeric_cols)}")
    
    # Compute statistics in batches
    batch_size = 1_000_000  # Process 1M rows at a time
    feature_sums = {col: 0.0 for col in numeric_cols}
    feature_sq_sums = {col: 0.0 for col in numeric_cols}
    feature_counts = {col: 0 for col in numeric_cols}
    total_rows = 0
    
    print(f"   Reading in batches of {batch_size:,} rows...")
    for batch_df in pl.read_csv_batched(train_file, batch_size=batch_size):
        total_rows += len(batch_df)
        for col in numeric_cols:
            col_data = batch_df[col].drop_nulls()
            if len(col_data) > 0:
                feature_sums[col] += float(col_data.sum())
                feature_sq_sums[col] += float((col_data ** 2).sum())
                feature_counts[col] += len(col_data)
        print(f"   Processed {total_rows:,} rows...", end='\r')
    
    print(f"\n✅ Statistics computed from {total_rows:,} rows in {time.time()-start:.1f}s")
    
    # Calculate mean and std
    feature_means = {}
    feature_stds = {}
    for col in numeric_cols:
        if feature_counts[col] > 0:
            feature_means[col] = feature_sums[col] / feature_counts[col]
            variance = (feature_sq_sums[col] / feature_counts[col]) - (feature_means[col] ** 2)
            feature_stds[col] = np.sqrt(max(0, variance))
            if feature_stds[col] == 0:
                feature_stds[col] = 1.0
        else:
            feature_means[col] = 0.0
            feature_stds[col] = 1.0
    
    # Save scaler
    scaler_data = {
        'mean': np.array([feature_means[c] for c in numeric_cols], dtype=np.float32),
        'std': np.array([feature_stds[c] for c in numeric_cols], dtype=np.float32),
        'feature_names': numeric_cols
    }
    scaler_path = os.path.join(output_dir, 'scaler.npz')
    np.savez(scaler_path, **scaler_data)
    print(f"💾 Saved scaler: {scaler_path}")
    
    # PASS 2: Read, group, normalize and save in chunks (memory efficient)
    print(f"\n📦 PASS 2: Creating customer sequences (streaming)...")
    
    chunk_idx = 0
    buffer = []
    total_sequences = 0
    current_customer = None
    customer_rows = []
    
    # Read in batches and process by customer
    for batch_df in pl.read_csv_batched(train_file, batch_size=batch_size):
        # Sort by customer and date
        batch_df = batch_df.sort(['customer_ID', 'S_2'])
        
        for row in batch_df.iter_rows(named=True):
            customer_id = row['customer_ID']
            
            # New customer - process previous customer's data
            if current_customer is not None and customer_id != current_customer:
                if len(customer_rows) >= 5:  # Skip short sequences
                    # Convert to array and normalize
                    seq_data = np.array([[row[col] for col in numeric_cols] for row in customer_rows], dtype=np.float32)
                    for i, col in enumerate(numeric_cols):
                        seq_data[:, i] = (seq_data[:, i] - feature_means[col]) / feature_stds[col]
                    seq_data = np.nan_to_num(seq_data, 0.0)
                    
                    buffer.append(seq_data)
                    total_sequences += 1
                    
                    # Save chunk when buffer is full
                    if len(buffer) >= chunk_size:
                        chunk_file = os.path.join(output_dir, f'chunk_{chunk_idx:04d}.npz')
                        np.savez_compressed(chunk_file, sequences=buffer)
                        print(f"  💾 Saved chunk {chunk_idx}: {len(buffer)} sequences")
                        chunk_idx += 1
                        buffer = []
                
                customer_rows = []
            
            current_customer = customer_id
            customer_rows.append(row)
    
    # Process last customer
    if len(customer_rows) >= 5:
        seq_data = np.array([[row[col] for col in numeric_cols] for row in customer_rows], dtype=np.float32)
        for i, col in enumerate(numeric_cols):
            seq_data[:, i] = (seq_data[:, i] - feature_means[col]) / feature_stds[col]
        seq_data = np.nan_to_num(seq_data, 0.0)
        buffer.append(seq_data)
        total_sequences += 1
    
    # Save remaining
    if buffer:
        chunk_file = os.path.join(output_dir, f'chunk_{chunk_idx:04d}.npz')
        np.savez_compressed(chunk_file, sequences=buffer)
        print(f"  💾 Saved chunk {chunk_idx}: {len(buffer)} sequences")
        chunk_idx += 1
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE")
    print(f"   Total sequences: {total_sequences:,}")
    print(f"   Total chunks: {chunk_idx}")
    print(f"   Features: {len(numeric_cols)}")
    print(f"   Output: {output_dir}")
    print("=" * 80)
    
    # Clean up raw CSV to save disk space
    print("🗑️  Removing raw CSV to save disk space...")
    try:
        os.remove(train_file)
        print(f"   Deleted: {train_file}")
    except Exception as e:
        print(f"   Warning: Could not delete {train_file}: {e}")
    
    return output_dir, len(numeric_cols)

# Training dataset
class ChunkedJEPADataset(torch.utils.data.Dataset):
    """Dataset that loads preprocessed chunks"""
    def __init__(self, chunk_dir, past_len=13, future_len=13, samples_per_chunk=200):
        self.chunk_files = sorted([
            os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) 
            if f.startswith('chunk_') and f.endswith('.npz')
        ])
        self.past_len = past_len
        self.future_len = future_len
        self.samples_per_chunk = samples_per_chunk
        
        print(f"[dataset] Found {len(self.chunk_files)} chunks")
        self.total_samples = len(self.chunk_files) * samples_per_chunk
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Pick random chunk
        chunk_idx = np.random.randint(0, len(self.chunk_files))
        data = np.load(self.chunk_files[chunk_idx], allow_pickle=True)
        sequences = data['sequences']
        
        # Pick random sequence
        seq_idx = np.random.randint(0, len(sequences))
        seq = sequences[seq_idx]
        
        # Pick random window
        min_len = self.past_len + self.future_len
        if seq.shape[0] < min_len:
            # Pad if needed
            pad_len = min_len - seq.shape[0]
            seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]), dtype=seq.dtype)])
        
        max_start = seq.shape[0] - min_len
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        past = seq[start_idx:start_idx + self.past_len]
        future = seq[start_idx + self.past_len:start_idx + self.past_len + self.future_len]
        
        return torch.from_numpy(past), torch.from_numpy(future)

def train_jepa(data_dir, cfg, args):
    """Train JEPA model on preprocessed data"""
    from src.models.encoder import build_encoder
    from src.models.predictor import MLPredictor
    from src.optim.ema import ema_update
    from src.optim.sched import cosine_warmup
    from src.loss import make_loss
    from src.data.augmentations import AugmentationPipeline
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎯 Training on {device}")
    
    # Load scaler info
    scaler_path = os.path.join(data_dir, 'scaler.npz')
    scaler_data = np.load(scaler_path)
    n_features = len(scaler_data['feature_names'])
    print(f"Features: {n_features}")
    
    # Create datasets
    train_ds = ChunkedJEPADataset(
        data_dir, 
        past_len=cfg["data"]["past_len"],
        future_len=cfg["data"]["future_len"],
        samples_per_chunk=200
    )
    
    val_ds = ChunkedJEPADataset(
        data_dir,
        past_len=cfg["data"]["past_len"],
        future_len=cfg["data"]["future_len"],
        samples_per_chunk=50
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    print("\n🏗️  Building model...")
    model_cfg = cfg["model"].copy()
    model_cfg["n_features"] = n_features
    
    enc = build_encoder(model_cfg).to(device)
    pred = MLPredictor(model_cfg["d_model"], n_features).to(device)
    
    enc_ema = build_encoder(model_cfg).to(device)
    enc_ema.load_state_dict(enc.state_dict())
    for p in enc_ema.parameters():
        p.requires_grad = False
    
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in pred.parameters())
    print(f"✅ Model: {n_params:,} parameters")
    
    # Optimizer
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(
        params, lr=cfg["train"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["train"].get("weight_decay", 1e-5)
    )
    
    loss_fn = make_loss()
    aug = AugmentationPipeline(
        time_mask_prob=cfg["model"]["masking"]["time_mask_prob"],
        feature_mask_prob=cfg["model"]["masking"]["feature_mask_prob"]
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("🚀 TRAINING")
    print("=" * 80)
    
    max_epochs = cfg["train"].get("max_epochs", 12)
    ema_decay = cfg["model"].get("ema_decay", 0.996)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    
    step = 0
    best_val_loss = float("inf")
    
    for epoch in range(max_epochs):
        enc.train()
        pred.train()
        epoch_loss = 0.0
        
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            
            past_aug, mask_t, mask_f = aug(past)
            z_past = enc(past_aug)
            z_future_target = enc_ema(future).detach()
            pred_future = pred(z_past)
            
            loss = loss_fn(pred_future, z_future_target, mask_t, mask_f)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            
            ema_update(enc, enc_ema, ema_decay)
            
            epoch_loss += loss.item()
            step += 1
        
        train_loss = epoch_loss / len(train_loader)
        
        # Validation
        enc.eval()
        pred.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                z_past = enc(past)
                z_future_target = enc_ema(future)
                pred_future = pred(z_past)
                loss = loss_fn(pred_future, z_future_target)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Report to SageMaker
        if args.sagemaker_mode:
            print(f"#quality_metric: name=validation:loss, iteration={epoch+1}, value={val_loss}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = args.model_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'encoder': enc.state_dict(),
                'encoder_ema': enc_ema.state_dict(),
                'predictor': pred.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, os.path.join(ckpt_dir, 'best.pt'))
            print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")
    
    print("\n✅ TRAINING COMPLETE")
    print(f"Best Val Loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/jepa_improved.yaml")
    parser.add_argument("--sagemaker-mode", type=int, default=0)
    parser.add_argument("--skip-download", type=int, default=0, help="Skip Kaggle download")
    parser.add_argument("--skip-preprocess", type=int, default=0, help="Skip preprocessing")
    
    # SageMaker hyperparameters
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
    parser.add_argument("--encoder", type=str, default="gru", help="Encoder type")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs (overrides config)")
    parser.add_argument("--mixed-precision", type=int, default=1, help="Use mixed precision")
    
    # Paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "checkpoints"))
    parser.add_argument("--data-dir", type=str, default="/tmp/amex_raw")
    parser.add_argument("--processed-dir", type=str, default="/tmp/processed")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 COMPLETE AWS JEPA TRAINING PIPELINE")
    print("=" * 80)
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Override with hyperparameters
    if args.learning_rate:
        cfg["train"]["lr"] = args.learning_rate
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.d_model:
        cfg["model"]["d_model"] = args.d_model
    if args.n_layers:
        cfg["model"]["n_layers"] = args.n_layers
    if args.dropout:
        cfg["model"]["dropout"] = args.dropout
    if args.ema_decay:
        cfg["model"]["ema_decay"] = args.ema_decay
    if args.weight_decay:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.grad_clip:
        cfg["train"]["grad_clip"] = args.grad_clip
    if args.time_mask_prob:
        cfg["model"]["masking"]["time_mask_prob"] = args.time_mask_prob
    if args.feature_mask_prob:
        cfg["model"]["masking"]["feature_mask_prob"] = args.feature_mask_prob
    
    # Step 1: Download data
    if not args.skip_download:
        raw_dir = download_amex_data(args.data_dir)
    else:
        raw_dir = args.data_dir
        print(f"⏭️  Skipping download, using: {raw_dir}")
    
    # Step 2: Preprocess
    if not args.skip_preprocess:
        processed_dir, n_features = preprocess_amex_data(raw_dir, args.processed_dir)
    else:
        processed_dir = args.processed_dir
        print(f"⏭️  Skipping preprocessing, using: {processed_dir}")
    
    # Step 3: Train
    train_jepa(processed_dir, cfg, args)
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
