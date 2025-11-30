#!/usr/bin/env python3
"""
Local inference with trained JEPA models
No AWS/SageMaker required - runs on your local machine
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Optional, Union, List
import warnings
warnings.filterwarnings('ignore')


class JEPAInference:
    """Local inference wrapper for JEPA encoder"""
    
    def __init__(self, model_dir: str, device: str = 'auto'):
        """
        Initialize JEPA model for local inference
        
        Args:
            model_dir: Path to directory containing encoder.pt and scaler.npz
            device: 'cpu', 'cuda', or 'auto' (auto-detect GPU)
        """
        self.model_dir = Path(model_dir)
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Load encoder
        print(f"📥 Loading encoder from {self.model_dir / 'encoder.pt'}...")
        self.encoder = torch.load(
            self.model_dir / 'encoder.pt',
            map_location=self.device
        )
        self.encoder.eval()
        
        # Load scaler
        print(f"📥 Loading scaler from {self.model_dir / 'scaler.npz'}...")
        scaler_data = np.load(self.model_dir / 'scaler.npz')
        self.scaler_mean = scaler_data['mean']
        self.scaler_std = scaler_data['std']
        
        # Load config if available
        config_path = self.model_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            print(f"📋 Model config: {self.config.get('d_model', 'unknown')}d, "
                  f"{self.config.get('n_layers', 'unknown')} layers")
        else:
            self.config = {}
        
        self.n_features = len(self.scaler_mean)
        print(f"✅ Model ready! Input features: {self.n_features}")
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize data using training scaler"""
        return (X - self.scaler_mean) / (self.scaler_std + 1e-8)
    
    def encode(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        batch_size: int = 32,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embeddings from financial data
        
        Args:
            data: Input data [seq_len, n_features] or [batch, seq_len, n_features]
            batch_size: Batch size for processing (if single sequence)
            return_numpy: Return numpy array (True) or torch tensor (False)
        
        Returns:
            embeddings: [seq_len, d_model] or [batch, seq_len, d_model]
        """
        # Convert to numpy
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Validate features
        if data.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {data.shape[-1]}. "
                f"Ensure your data has the same features as training."
            )
        
        # Add batch dimension if needed
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # [1, seq_len, features]
        
        # Normalize
        data_norm = self.normalize(data)
        
        # Convert to tensor
        X = torch.FloatTensor(data_norm).to(self.device)
        
        # Inference
        with torch.no_grad():
            embeddings = self.encoder(X)
        
        if return_numpy:
            embeddings = embeddings.cpu().numpy()
        
        return embeddings.squeeze(0) if embeddings.shape[0] == 1 else embeddings
    
    def encode_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Encode a pandas DataFrame
        
        Args:
            df: DataFrame with features
            feature_cols: List of column names to use (None = all columns)
        
        Returns:
            embeddings: [seq_len, d_model]
        """
        if feature_cols is not None:
            data = df[feature_cols].values
        else:
            data = df.values
        
        return self.encode(data)
    
    def encode_csv(self, csv_path: str, **kwargs) -> np.ndarray:
        """Encode data from CSV file"""
        df = pd.read_csv(csv_path)
        return self.encode_dataframe(df, **kwargs)
    
    def get_sequence_embedding(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Get single embedding vector for entire sequence
        
        Args:
            data: Input sequence
            pooling: 'mean', 'max', 'first', or 'last'
        
        Returns:
            embedding: [d_model]
        """
        embeddings = self.encode(data)  # [seq_len, d_model]
        
        if pooling == 'mean':
            return embeddings.mean(axis=0)
        elif pooling == 'max':
            return embeddings.max(axis=0)
        elif pooling == 'first':
            return embeddings[0]
        elif pooling == 'last':
            return embeddings[-1]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    
    def batch_encode(
        self,
        data_list: List[Union[np.ndarray, pd.DataFrame]],
        pooling: Optional[str] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple sequences
        
        Args:
            data_list: List of sequences
            pooling: If specified, return pooled embeddings [batch, d_model]
                    If None, return full sequences (requires same length)
            show_progress: Show progress bar
        
        Returns:
            embeddings: [batch, seq_len, d_model] or [batch, d_model]
        """
        from tqdm import tqdm
        
        embeddings = []
        iterator = tqdm(data_list, desc="Encoding") if show_progress else data_list
        
        for data in iterator:
            if pooling:
                emb = self.get_sequence_embedding(data, pooling=pooling)
            else:
                emb = self.encode(data)
            embeddings.append(emb)
        
        return np.stack(embeddings)


def demo_feature_extraction():
    """Demo: Extract features from financial data"""
    print("\n" + "="*60)
    print("DEMO: Feature Extraction")
    print("="*60 + "\n")
    
    # Initialize model
    model = JEPAInference('models/job8_best')
    
    # Create dummy data (replace with your real data)
    print("📊 Creating sample transaction data...")
    n_transactions = 100
    n_features = model.n_features
    
    # Simulate customer transaction history
    sample_data = np.random.randn(n_transactions, n_features)
    
    # Extract embeddings
    print(f"🔄 Encoding {n_transactions} transactions...")
    embeddings = model.encode(sample_data)
    
    print(f"✅ Output shape: {embeddings.shape}")
    print(f"   Each transaction → {embeddings.shape[1]}-dimensional vector")
    
    # Get overall customer embedding
    customer_embedding = model.get_sequence_embedding(sample_data, pooling='mean')
    print(f"\n📊 Customer profile embedding: {customer_embedding.shape}")
    
    return model, embeddings


def demo_fraud_detection():
    """Demo: Fraud detection with fine-tuning"""
    print("\n" + "="*60)
    print("DEMO: Fraud Detection")
    print("="*60 + "\n")
    
    import torch.nn as nn
    import torch.optim as optim
    
    # Load model
    model = JEPAInference('models/job8_best')
    
    # Create fraud detection classifier
    class FraudDetector(nn.Module):
        def __init__(self, encoder, d_model):
            super().__init__()
            self.encoder = encoder
            
            # Freeze encoder (optional - set requires_grad=True to fine-tune)
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # x: [batch, seq_len, features]
            features = self.encoder(x)  # [batch, seq_len, d_model]
            pooled = features.mean(dim=1)  # [batch, d_model]
            return self.classifier(pooled)  # [batch, 1]
    
    d_model = model.config.get('d_model', 212)
    fraud_model = FraudDetector(model.encoder, d_model).to(model.device)
    
    print(f"✅ Fraud detector created with {d_model}d encoder")
    print("   Encoder frozen, only training classification head")
    print("\nTo train:")
    print("  criterion = nn.BCELoss()")
    print("  optimizer = optim.Adam(fraud_model.classifier.parameters(), lr=0.001)")
    print("  # Then standard training loop with your labeled fraud data")
    
    return fraud_model


def demo_clustering():
    """Demo: Customer segmentation via clustering"""
    print("\n" + "="*60)
    print("DEMO: Customer Segmentation")
    print("="*60 + "\n")
    
    from sklearn.cluster import KMeans
    
    # Load model
    model = JEPAInference('models/job8_best')
    
    # Simulate multiple customers
    print("📊 Creating 500 customer profiles...")
    n_customers = 500
    customer_embeddings = []
    
    for i in range(n_customers):
        # Each customer has variable-length transaction history
        n_trans = np.random.randint(20, 200)
        transactions = np.random.randn(n_trans, model.n_features)
        
        # Get customer embedding
        emb = model.get_sequence_embedding(transactions, pooling='mean')
        customer_embeddings.append(emb)
    
    X = np.stack(customer_embeddings)  # [500, d_model]
    
    # Cluster customers
    print("🔄 Clustering into 5 segments...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Analyze clusters
    print("\n📊 Customer Segments:")
    for i in range(5):
        count = (clusters == i).sum()
        pct = 100 * count / n_customers
        print(f"  Segment {i+1}: {count} customers ({pct:.1f}%)")
    
    return X, clusters


def demo_anomaly_detection():
    """Demo: Detect unusual transactions"""
    print("\n" + "="*60)
    print("DEMO: Anomaly Detection")
    print("="*60 + "\n")
    
    from sklearn.ensemble import IsolationForest
    
    # Load model
    model = JEPAInference('models/job8_best')
    
    # Create normal transaction embeddings
    print("📊 Training on normal transactions...")
    normal_embeddings = []
    for _ in range(1000):
        # Normal transaction
        trans = np.random.randn(1, model.n_features) * 0.5  # Low variance
        emb = model.encode(trans).squeeze()
        normal_embeddings.append(emb)
    
    X_normal = np.stack(normal_embeddings)
    
    # Train anomaly detector
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_normal)
    
    print("✅ Anomaly detector trained")
    
    # Test on new transactions
    print("\n🔍 Testing on new transactions...")
    test_transactions = [
        np.random.randn(1, model.n_features) * 0.5,  # Normal
        np.random.randn(1, model.n_features) * 3.0,  # Anomaly (high variance)
        np.random.randn(1, model.n_features) * 0.6,  # Normal
    ]
    
    for i, trans in enumerate(test_transactions, 1):
        emb = model.encode(trans).squeeze()
        prediction = iso_forest.predict([emb])[0]
        score = iso_forest.score_samples([emb])[0]
        
        status = "✅ NORMAL" if prediction == 1 else "🚨 ANOMALY"
        print(f"  Transaction {i}: {status} (score: {score:.3f})")
    
    return iso_forest


if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("🚀 JEPA Model - Local Inference")
    print("="*60)
    
    # Check if model exists
    model_dir = Path('models/job8_best')
    if not model_dir.exists():
        print(f"\n❌ Model not found at {model_dir}")
        print("\nFirst, download the model:")
        print("  cd /mnt/e/JEPA/aws-financial-jepa/scripts")
        print("  ./download_job8_model.sh")
        sys.exit(1)
    
    # Run demos
    try:
        # Feature extraction
        model, embeddings = demo_feature_extraction()
        
        # Fraud detection
        fraud_model = demo_fraud_detection()
        
        # Clustering
        X, clusters = demo_clustering()
        
        # Anomaly detection
        iso_forest = demo_anomaly_detection()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print("  1. Replace dummy data with your real financial data")
        print("  2. Use model.encode(data) for feature extraction")
        print("  3. Build downstream models (fraud, risk, etc.)")
        print("  4. See HOW_TO_USE_MODELS.md for more examples")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
