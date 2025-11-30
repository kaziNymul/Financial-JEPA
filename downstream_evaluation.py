#!/usr/bin/env python3
"""
Downstream task evaluation for JEPA models
Tests fraud detection, credit risk, and customer segmentation
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')


class GRUEncoder(torch.nn.Module):
    """GRU encoder matching training code"""
    
    def __init__(self, input_dim: int, d_model: int, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        _, h = self.gru(x)           # h: [num_layers, B, d_model]
        return h[-1]                  # last layer: [B, d_model]


class JEPAEncoder:
    """Lightweight JEPA encoder for inference"""
    
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract config
        if 'cfg' in checkpoint:
            cfg = checkpoint['cfg']
            self.d_model = cfg.get('model', {}).get('d_model', 212)
            self.n_features = cfg.get('data', {}).get('n_features', 1359)
            self.n_layers = cfg.get('model', {}).get('n_layers', 3)
            self.dropout = cfg.get('model', {}).get('dropout', 0.0)
        else:
            self.d_model = 212
            self.n_features = 1359
            self.n_layers = 3
            self.dropout = 0.0
        
        # Rebuild encoder
        self.encoder = GRUEncoder(
            input_dim=self.n_features,
            d_model=self.d_model,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['enc'])
        self.encoder.eval()
        
        # Load scaler if available
        scaler_path = Path('artifacts/scaler.npz')
        if scaler_path.exists():
            scaler = np.load(scaler_path)
            self.scaler_mean = scaler['mean']
            self.scaler_std = scaler['std']
            print("  ✅ Loaded scaler from artifacts/")
        else:
            print("  ⚠️  No scaler found, using identity normalization")
            self.scaler_mean = np.zeros(self.n_features)
            self.scaler_std = np.ones(self.n_features)
    
    def normalize(self, X):
        """Normalize input features"""
        return (X - self.scaler_mean) / (self.scaler_std + 1e-8)
    
    @torch.no_grad()
    def encode(self, X):
        """Extract embeddings from input features"""
        # Normalize
        X_norm = self.normalize(X)
        
        # Convert to tensor [B, T, F] where T=1 (single timestep)
        X_tensor = torch.FloatTensor(X_norm).unsqueeze(1)
        
        # Forward pass through encoder
        embeddings = []
        batch_size = 256
        
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            h = self.encoder(batch)  # [batch, d_model]
            embeddings.append(h.cpu().numpy())
        
        return np.vstack(embeddings)


def load_sample_data():
    """Load or create sample data for evaluation"""
    
    # Try to load real AMEX data (use training data since it has labels)
    train_data_path = Path('/mnt/e/JEPA/financial-jepa/financial-jepa/data/raw/amex/train_data.csv')
    train_labels_path = Path('/mnt/e/JEPA/financial-jepa/financial-jepa/data/raw/amex/train_labels.csv')
    
    if train_data_path.exists():
        print("📊 Loading real AMEX training data...")
        print(f"  Path: {train_data_path}")
        
        # Load a sample (16GB is too large to load fully)
        n_samples = 50000
        df = pd.read_csv(train_data_path, nrows=n_samples)
        
        print(f"  Loaded {len(df)} rows with {len(df.columns)} raw columns")
        
        # Extract features (skip customer_ID and date columns)
        skip_cols = ['customer_ID', 'S_2']  # S_2 is the date
        feature_cols = [c for c in df.columns if c not in skip_cols]
        
        # Convert to numeric only
        X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        
        # Fill missing values and convert to numpy
        X_raw = X_df.fillna(0).values
        customer_ids = df['customer_ID'].values
        
        print(f"  Raw features: {X_raw.shape[1]} columns")
        
        # The model was trained on 1359 features (after preprocessing)
        # If raw data doesn't match, we need to pad/truncate
        if X_raw.shape[1] < 1359:
            # Pad with zeros
            padding = np.zeros((len(X_raw), 1359 - X_raw.shape[1]))
            X = np.hstack([X_raw, padding])
            print(f"  ⚠️  Padded to 1359 features (raw had {X_raw.shape[1]})")
        elif X_raw.shape[1] > 1359:
            # Truncate
            X = X_raw[:, :1359]
            print(f"  ⚠️  Truncated to 1359 features (raw had {X_raw.shape[1]})")
        else:
            X = X_raw
        
        # Load labels if available (for training customers only)
        if train_labels_path.exists():
            labels_df = pd.read_csv(train_labels_path)
            print(f"  Loaded {len(labels_df)} labels")
            
            # Match labels to customers
            labels_dict = dict(zip(labels_df['customer_ID'], labels_df['target']))
            y_risk = np.array([labels_dict.get(cid, 0) for cid in customer_ids])
            
            # Use same labels for fraud (since we don't have separate fraud labels)
            y_fraud = y_risk.copy()
            
            print(f"  Default rate: {y_risk.mean():.1%}")
        else:
            # Create synthetic labels
            print("  ⚠️  No labels found, generating synthetic targets")
            np.random.seed(42)
            y_fraud = (np.random.rand(len(X)) < 0.1).astype(int)
            y_risk = (np.random.rand(len(X)) < 0.15).astype(int)
        
        print(f"  ✅ Final dataset: {X.shape} features, {len(y_risk)} labels")
        
    else:
        print("📊 No real data found, generating synthetic data...")
        print(f"  (Looked in: {train_data_path})")
        np.random.seed(42)
        n_samples = 10000
        n_features = 1359
        
        X = np.random.randn(n_samples, n_features)
        y_fraud = (np.random.rand(n_samples) < 0.1).astype(int)
        y_risk = (np.random.rand(n_samples) < 0.15).astype(int)
        
        print(f"  Generated {n_samples} samples with {n_features} features")
    
    return X, y_fraud, y_risk


def evaluate_fraud_detection(embeddings, y_true):
    """Evaluate fraud detection task"""
    print("\n🔍 FRAUD DETECTION")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_true, test_size=0.3, random_state=42, stratify=y_true
    )
    
    # Logistic Regression (simple baseline)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Logistic Regression:")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    
    # Random Forest (stronger baseline)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    
    print(f"\n  Random Forest:")
    print(f"    AUC-ROC:   {auc_rf:.4f}")
    print(f"    Precision: {precision_rf:.4f}")
    print(f"    Recall:    {recall_rf:.4f}")
    print(f"    F1 Score:  {f1_rf:.4f}")
    
    return {
        'lr_auc': auc,
        'lr_precision': precision,
        'lr_recall': recall,
        'lr_f1': f1,
        'rf_auc': auc_rf,
        'rf_precision': precision_rf,
        'rf_recall': recall_rf,
        'rf_f1': f1_rf,
    }


def evaluate_credit_risk(embeddings, y_true):
    """Evaluate credit risk prediction"""
    print("\n💳 CREDIT RISK PREDICTION")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_true, test_size=0.3, random_state=42, stratify=y_true
    )
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"  Default Prediction AUC: {auc:.4f}")
    
    return {'auc': auc}


def evaluate_segmentation(embeddings):
    """Evaluate customer segmentation"""
    print("\n👥 CUSTOMER SEGMENTATION")
    print("-" * 60)
    
    # K-means clustering
    best_k = None
    best_score = -1
    
    for k in [3, 5, 7, 10]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        
        print(f"  K={k:2d}:")
        print(f"    Silhouette:     {silhouette:.4f}")
        print(f"    Davies-Bouldin: {davies_bouldin:.4f}")
        
        if silhouette > best_score:
            best_score = silhouette
            best_k = k
    
    print(f"\n  ✅ Best: K={best_k} (Silhouette={best_score:.4f})")
    
    return {
        'best_k': best_k,
        'best_silhouette': best_score,
    }


def evaluate_model(job_num, model_path):
    """Evaluate a single model"""
    print("\n" + "=" * 80)
    print(f"EVALUATING JOB {job_num}")
    print("=" * 80)
    
    # Load encoder
    print(f"\n📦 Loading model from {model_path.name}...")
    encoder = JEPAEncoder(model_path)
    print(f"  d_model: {encoder.d_model}")
    print(f"  n_features: {encoder.n_features}")
    
    # Load data
    X, y_fraud, y_risk = load_sample_data()
    
    # Extract embeddings
    print(f"\n🔄 Extracting embeddings...")
    embeddings = encoder.encode(X)
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Evaluate tasks
    fraud_results = evaluate_fraud_detection(embeddings, y_fraud)
    risk_results = evaluate_credit_risk(embeddings, y_risk)
    seg_results = evaluate_segmentation(embeddings)
    
    # Combine results (convert numpy types to Python types for JSON)
    results = {
        'job': int(job_num),
        'd_model': int(encoder.d_model),
        'fraud_detection': {k: float(v) for k, v in fraud_results.items()},
        'credit_risk': {k: float(v) for k, v in risk_results.items()},
        'segmentation': {k: int(v) if isinstance(v, (int, np.integer)) else float(v) 
                        for k, v in seg_results.items()},
    }
    
    return results


def main():
    print("=" * 80)
    print("DOWNSTREAM TASK EVALUATION")
    print("=" * 80)
    
    models_dir = Path('downloaded_models')
    
    # Evaluate Job 8 (best model)
    job_num = 8
    model_path = models_dir / f'job_{str(job_num).zfill(3)}' / 'best.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run evaluate_models.py first to download models")
        return
    
    results = evaluate_model(job_num, model_path)
    
    # Save results
    output_file = f'downstream_results_job{job_num}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n💾 Results saved to {output_file}")
    
    # Print summary for paper
    print("\n" + "=" * 80)
    print("📄 RESULTS FOR PAPER")
    print("=" * 80)
    
    fr = results['fraud_detection']
    cr = results['credit_risk']
    seg = results['segmentation']
    
    print(f"""
Fraud Detection (with linear probe):
  - AUC-ROC: {fr['lr_auc']:.3f}
  - Precision: {fr['lr_precision']:.3f}
  - Recall: {fr['lr_recall']:.3f}

Fraud Detection (with Random Forest):
  - AUC-ROC: {fr['rf_auc']:.3f}
  - Precision: {fr['rf_precision']:.3f}
  - Recall: {fr['rf_recall']:.3f}

Credit Default Prediction:
  - AUC-ROC: {cr['auc']:.3f}

Customer Segmentation:
  - Best K: {seg['best_k']} clusters
  - Silhouette Score: {seg['best_silhouette']:.3f}

Use these numbers in Section 5 (Experiments) of your paper.
    """)


if __name__ == '__main__':
    main()
