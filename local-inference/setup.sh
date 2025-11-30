#!/bin/bash
# Setup local inference environment

set -e

echo "=========================================="
echo "🔧 SETTING UP LOCAL INFERENCE"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅ Dependencies installed:"
pip list | grep -E "torch|numpy|pandas|scikit-learn|boto3"

# Download model if needed
echo ""
MODEL_DIR="models/job8_best"
if [ ! -d "$MODEL_DIR" ]; then
    echo "📥 Model not found. Downloading..."
    
    # Create directory
    mkdir -p models
    
    # Download from S3
    echo "   Downloading from S3..."
    aws s3 cp s3://financial-jepa-data-057149785966/models/jepa-20251130-041149-008-dbd83087/output/model.tar.gz \
        models/model.tar.gz --region us-east-1
    
    # Extract
    echo "   Extracting..."
    mkdir -p "$MODEL_DIR"
    tar -xzf models/model.tar.gz -C "$MODEL_DIR"
    rm models/model.tar.gz
    
    echo "   ✅ Model downloaded to $MODEL_DIR"
else
    echo "✅ Model found at $MODEL_DIR"
fi

echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Model location: $MODEL_DIR"
echo "Model files:"
ls -lh "$MODEL_DIR"
echo ""
echo "To run demos:"
echo "  source venv/bin/activate"
echo "  python inference.py"
echo ""
echo "To use in your code:"
echo "  from inference import JEPAInference"
echo "  model = JEPAInference('$MODEL_DIR')"
echo "  embeddings = model.encode(your_data)"
echo ""
