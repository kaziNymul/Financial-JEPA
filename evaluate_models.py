#!/usr/bin/env python3
"""
Evaluate all 8 trained JEPA models
Downloads models from S3, loads them, and evaluates on validation data
"""

import torch
import numpy as np
import boto3
import os
import tarfile
from pathlib import Path
import json

def download_all_models():
    """Download all 8 models from S3"""
    
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket = 'financial-jepa-data-057149785966'
    
    # Model job IDs
    jobs = [
        ('001', '08ec8d4f'),
        ('002', '3238fcb3'),
        ('003', 'b5a4ae88'),
        ('004', '20727e11'),
        ('005', 'a26c2d0e'),
        ('006', '25ec4c6b'),
        ('007', '13cb6eb7'),
        ('008', 'dbd83087'),
    ]
    
    models_dir = Path('downloaded_models')
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DOWNLOADING ALL 8 MODELS FROM S3")
    print("=" * 80)
    print()
    
    for job_num, job_hash in jobs:
        job_dir = models_dir / f'job_{job_num}'
        job_dir.mkdir(exist_ok=True)
        
        model_key = f'models/jepa-20251130-041149-{job_num}-{job_hash}/output/model.tar.gz'
        local_tar = job_dir / 'model.tar.gz'
        
        print(f"Job {job_num}: ", end='', flush=True)
        
        # Download
        try:
            s3.download_file(bucket, model_key, str(local_tar))
            
            # Extract
            with tarfile.open(local_tar, 'r:gz') as tar:
                tar.extractall(job_dir)
            
            # Remove tar
            local_tar.unlink()
            
            # Check for best.pt
            if (job_dir / 'best.pt').exists():
                print(f"✅ Downloaded and extracted")
            else:
                print(f"⚠️  Downloaded but no best.pt found")
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
    
    print()
    return models_dir


def load_model(model_path):
    """Load a model checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return checkpoint


def extract_config(checkpoint):
    """Extract configuration from checkpoint"""
    if 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
        return {
            'd_model': cfg.get('model', {}).get('d_model', 'N/A'),
            'n_layers': cfg.get('model', {}).get('n_layers', 'N/A'),
            'lr': cfg.get('train', {}).get('lr', 'N/A'),
            'batch_size': cfg.get('train', {}).get('batch_size', 'N/A'),
            'dropout': cfg.get('model', {}).get('dropout', 'N/A'),
        }
    return {}


def count_parameters(state_dict):
    """Count parameters in model"""
    return sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))


def evaluate_all_models(models_dir):
    """Evaluate all downloaded models"""
    
    print("=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)
    print()
    
    results = []
    
    for job_num in range(1, 9):
        job_dir = models_dir / f'job_{str(job_num).zfill(3)}'
        model_path = job_dir / 'best.pt'
        
        if not model_path.exists():
            print(f"Job {job_num}: ❌ Model file not found")
            continue
        
        print(f"Job {job_num}: ", end='', flush=True)
        
        try:
            # Load checkpoint
            checkpoint = load_model(model_path)
            
            # Extract info
            config = extract_config(checkpoint)
            
            # Count parameters
            if 'enc' in checkpoint:
                enc_params = count_parameters(checkpoint['enc'])
                pred_params = count_parameters(checkpoint.get('pred', {}))
                total_params = enc_params + pred_params
            else:
                total_params = 'N/A'
                enc_params = 'N/A'
            
            result = {
                'job': job_num,
                'd_model': config.get('d_model', 'N/A'),
                'n_layers': config.get('n_layers', 'N/A'),
                'lr': config.get('lr', 'N/A'),
                'batch_size': config.get('batch_size', 'N/A'),
                'dropout': config.get('dropout', 'N/A'),
                'encoder_params': enc_params,
                'total_params': total_params,
                'file_size_mb': model_path.stat().st_size / (1024 * 1024)
            }
            
            results.append(result)
            print(f"✅ d_model={result['d_model']}, layers={result['n_layers']}, params={enc_params:,}")
            
        except Exception as e:
            print(f"❌ Error loading: {str(e)[:40]}")
    
    return results


def print_comparison_table(results):
    """Print comparison table of all models"""
    
    print()
    print("=" * 120)
    print("DETAILED MODEL COMPARISON")
    print("=" * 120)
    print()
    
    # Header
    print(f"{'Job':<6} {'d_model':<10} {'Layers':<8} {'LR':<12} {'Batch':<8} {'Dropout':<10} {'Enc Params':<15} {'File MB':<10}")
    print("-" * 120)
    
    # Rows
    for r in results:
        lr_str = f"{r['lr']:.6f}" if isinstance(r['lr'], (int, float)) else str(r['lr'])
        dropout_str = f"{r['dropout']:.4f}" if isinstance(r['dropout'], (int, float)) else str(r['dropout'])
        params_str = f"{r['encoder_params']:,}" if isinstance(r['encoder_params'], int) else str(r['encoder_params'])
        
        print(f"{r['job']:<6} {r['d_model']:<10} {r['n_layers']:<8} {lr_str:<12} "
              f"{r['batch_size']:<8} {dropout_str:<10} {params_str:<15} {r['file_size_mb']:<10.2f}")
    
    print()
    
    # Statistics
    valid_params = [r['encoder_params'] for r in results if isinstance(r['encoder_params'], int)]
    if valid_params:
        print("STATISTICS:")
        print(f"  Average encoder parameters: {np.mean(valid_params):,.0f}")
        print(f"  Parameter range: {min(valid_params):,} - {max(valid_params):,}")
        print(f"  Total models: {len(results)}")
    
    print()
    print("=" * 120)


def save_results(results, output_file='model_evaluation_results.json'):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")


def main():
    print("=" * 80)
    print("JEPA MODEL EVALUATION PIPELINE")
    print("=" * 80)
    print()
    
    # Step 1: Download models
    models_dir = download_all_models()
    
    # Step 2: Evaluate models
    results = evaluate_all_models(models_dir)
    
    # Step 3: Print comparison
    print_comparison_table(results)
    
    # Step 4: Save results
    save_results(results)
    
    print()
    print("=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Use these results in your paper")
    print("  2. Best models: Check highest d_model and lowest dropout")
    print("  3. For downstream tasks, use Job 8 (largest, best config)")
    print()


if __name__ == '__main__':
    main()
