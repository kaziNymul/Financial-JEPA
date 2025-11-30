#!/bin/bash
# Download best trained model from S3

set -e

echo "=========================================="
echo "📥 DOWNLOADING BEST MODEL"
echo "=========================================="

# Get configuration
cd "$(dirname "$0")/../infra"
S3_BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

if [ -z "$S3_BUCKET" ]; then
    echo "❌ Error: Could not get S3 bucket from Terraform"
    exit 1
fi

# Get job name
if [ -f "../scripts/current_training_job.txt" ]; then
    JOB_NAME=$(cat ../scripts/current_training_job.txt)
else
    echo "❌ No training job found"
    echo "Please provide job name as argument"
    exit 1
fi

if [ ! -z "$1" ]; then
    JOB_NAME="$1"
fi

echo ""
echo "Job: $JOB_NAME"
echo "Bucket: $S3_BUCKET"
echo ""

# Get best training job
echo "🔍 Finding best model..."
BEST_JOB=$(aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job \
    --hyper-parameter-tuning-job-name "$JOB_NAME" \
    --sort-by FinalObjectiveMetricValue \
    --max-results 1 \
    --region $AWS_REGION \
    --query 'TrainingJobSummaries[0].TrainingJobName' \
    --output text)

if [ -z "$BEST_JOB" ] || [ "$BEST_JOB" == "None" ]; then
    echo "❌ No completed training jobs found"
    exit 1
fi

echo "✅ Best job: $BEST_JOB"

# Get model S3 path
MODEL_PATH=$(aws sagemaker describe-training-job \
    --training-job-name "$BEST_JOB" \
    --region $AWS_REGION \
    --query 'ModelArtifacts.S3ModelArtifacts' \
    --output text)

echo "📦 Model location: $MODEL_PATH"

# Download model
OUTPUT_DIR="../../financial-jepa/financial-jepa/trained_models"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📥 Downloading model..."
aws s3 cp "$MODEL_PATH" "$OUTPUT_DIR/model.tar.gz" --region $AWS_REGION

# Extract model
cd "$OUTPUT_DIR"
tar -xzf model.tar.gz
rm model.tar.gz

echo ""
echo "=========================================="
echo "✅ MODEL DOWNLOADED"
echo "=========================================="
echo ""
echo "📂 Location: $OUTPUT_DIR"
echo "🏆 Best job: $BEST_JOB"
echo ""
echo "Model files:"
ls -lh
echo ""
