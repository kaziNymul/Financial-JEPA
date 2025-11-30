#!/bin/bash
# Download Job 8 (best model) from S3

set -e

echo "=========================================="
echo "📥 DOWNLOADING JOB 8 MODEL (BEST)"
echo "=========================================="
echo ""
echo "Job 8 Performance:"
echo "  - Validation Loss: 0.096 (BEST)"
echo "  - Architecture: 3 layers, d_model=212"
echo "  - Learning Rate: 0.000566"
echo "  - Batch Size: 88"
echo ""

# S3 paths for Job 8
S3_PATH="s3://financial-jepa-data-057149785966/models/jepa-20251130-041149-008-dbd83087/output/model.tar.gz"
OUTPUT_DIR="/mnt/e/JEPA/financial-jepa/financial-jepa/trained_models/job8_best"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📥 Downloading from S3..."
aws s3 cp "$S3_PATH" "$OUTPUT_DIR/model.tar.gz" --region us-east-1

# Extract
cd "$OUTPUT_DIR"
echo "📦 Extracting model files..."
tar -xzf model.tar.gz
rm model.tar.gz

echo ""
echo "=========================================="
echo "✅ MODEL DOWNLOADED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "📂 Location: $OUTPUT_DIR"
echo ""
echo "Model files:"
ls -lh
echo ""
echo "Next steps:"
echo "1. See HOW_TO_USE_MODELS.md for usage examples"
echo "2. Load encoder: torch.load('$OUTPUT_DIR/encoder.pt')"
echo "3. Load scaler: np.load('$OUTPUT_DIR/scaler.npz')"
echo ""
