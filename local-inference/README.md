# Local JEPA Inference

Run trained JEPA models on your local machine without AWS/SageMaker.

## Quick Start

### 1. Setup Environment

```bash
cd /mnt/e/JEPA/local-inference
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environment
- Install dependencies (PyTorch, NumPy, pandas, scikit-learn)
- Download Job 8 model from S3
- Extract model files

### 2. Run Demos

```bash
source venv/bin/activate
python inference.py
```

This runs 4 demos:
1. **Feature Extraction**: Convert transactions → embeddings
2. **Fraud Detection**: Classification with frozen encoder
3. **Customer Segmentation**: K-means clustering on embeddings
4. **Anomaly Detection**: Isolation Forest for unusual transactions

## Usage in Your Code

### Basic Feature Extraction

```python
from inference import JEPAInference
import pandas as pd

# Load model
model = JEPAInference('models/job8_best')

# Load your data (must have 1359 features)
df = pd.read_csv('customer_transactions.csv')

# Extract embeddings
embeddings = model.encode(df)  # [seq_len, 212]

# Get single customer embedding
customer_profile = model.get_sequence_embedding(df, pooling='mean')  # [212]
```

### Batch Processing

```python
# Process multiple customers
customer_files = ['customer_1.csv', 'customer_2.csv', ...]

customer_embeddings = []
for file in customer_files:
    df = pd.read_csv(file)
    emb = model.get_sequence_embedding(df, pooling='mean')
    customer_embeddings.append(emb)

# Stack into matrix
X = np.stack(customer_embeddings)  # [n_customers, 212]

# Now use for clustering, classification, etc.
```

### Transfer Learning Example

```python
import torch
import torch.nn as nn

# Load JEPA encoder
model = JEPAInference('models/job8_best')

# Create task-specific model
class RiskPredictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Risk prediction head
        self.head = nn.Sequential(
            nn.Linear(212, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x).mean(dim=1)  # Pool sequence
        return self.head(features)

# Create model
risk_model = RiskPredictor(model.encoder)

# Train on your labeled data
optimizer = torch.optim.Adam(risk_model.head.parameters(), lr=0.001)
# ... training loop ...
```

## File Structure

```
local-inference/
├── setup.sh              # Setup script
├── requirements.txt      # Python dependencies
├── inference.py          # Main inference code + demos
├── README.md            # This file
├── venv/                # Virtual environment (created by setup)
└── models/
    └── job8_best/       # Downloaded model
        ├── encoder.pt   # JEPA encoder (212d)
        ├── scaler.npz   # Normalization params
        └── config.json  # Hyperparameters
```

## Model Specifications

**Job 8 (Best Model)**:
- Validation Loss: 0.096
- Architecture: 3 layers, 212-dimensional embeddings
- Learning Rate: 0.000566
- Batch Size: 88
- Input: 1359 features
- Output: 212-dimensional embeddings

## Requirements

- Python 3.8+
- CPU or GPU (auto-detected)
- 4GB RAM minimum
- 1GB disk space for model

## GPU vs CPU

The model automatically detects and uses GPU if available:

```python
# Force CPU
model = JEPAInference('models/job8_best', device='cpu')

# Force GPU
model = JEPAInference('models/job8_best', device='cuda')

# Auto-detect (default)
model = JEPAInference('models/job8_best', device='auto')
```

**Performance**:
- GPU (T4): ~1000 sequences/sec
- CPU: ~50-100 sequences/sec

## Common Use Cases

### 1. Credit Risk Scoring

```python
# Extract features from application data
features = model.encode(applicant_transactions)

# Train/use risk model
risk_score = risk_model(features)
```

### 2. Fraud Detection

```python
# Real-time fraud check
transaction_emb = model.encode(new_transaction)
is_fraud = fraud_classifier(transaction_emb)
```

### 3. Customer Segmentation

```python
# Cluster similar customers
embeddings = model.batch_encode(customer_data, pooling='mean')
segments = KMeans(n_clusters=5).fit_predict(embeddings)
```

### 4. Recommendation System

```python
# Find similar customers
query_emb = model.get_sequence_embedding(customer_A)
all_embs = model.batch_encode(all_customers, pooling='mean')

# Cosine similarity
similarities = cosine_similarity([query_emb], all_embs)[0]
similar_customers = np.argsort(similarities)[-10:]  # Top 10
```

## Troubleshooting

### Model not found
```bash
# Download manually
cd models
aws s3 cp s3://financial-jepa-data-057149785966/models/jepa-20251130-041149-008-dbd83087/output/model.tar.gz .
tar -xzf model.tar.gz -C job8_best/
```

### Out of memory
```python
# Use CPU instead of GPU
model = JEPAInference('models/job8_best', device='cpu')

# Or process in smaller batches
for batch in chunked_data:
    embeddings = model.encode(batch)
```

### Wrong feature count
```python
# Your data must have exactly 1359 features
# Check feature list used during training
print(f"Expected features: {model.n_features}")
print(f"Your data features: {your_data.shape[-1]}")

# Ensure same features in same order as training
```

## Performance Tips

1. **Batch Processing**: Process multiple sequences together
2. **GPU Usage**: Use GPU for large-scale inference
3. **TorchScript**: Compile model for 2-3x speedup
   ```python
   encoder_jit = torch.jit.script(model.encoder)
   torch.jit.save(encoder_jit, 'encoder_jit.pt')
   ```
4. **ONNX Export**: Convert to ONNX for cross-platform deployment

## Next Steps

1. Replace demo data with your real financial data
2. Train downstream models for your specific tasks
3. Deploy to production (see ../aws-financial-jepa/HOW_TO_USE_MODELS.md)
4. Fine-tune encoder if needed (unfreeze weights)

## Support

For issues or questions:
1. Check HOW_TO_USE_MODELS.md in parent directory
2. Verify data has correct 1359 features
3. Ensure model downloaded completely
4. Try CPU if GPU issues occur
