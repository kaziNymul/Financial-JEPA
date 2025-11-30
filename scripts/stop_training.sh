#!/bin/bash
# Stop training job immediately

set -e

SCRIPT_DIR="$(dirname "$0")"

# Get job name
if [ ! -z "$1" ]; then
    JOB_NAME="$1"
elif [ -f "$SCRIPT_DIR/../infra/current_training_job.txt" ]; then
    JOB_NAME=$(cat "$SCRIPT_DIR/../infra/current_training_job.txt")
else
    echo "❌ No training job found"
    echo "Usage: $0 [job-name]"
    exit 1
fi

cd "$SCRIPT_DIR/../infra"
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

echo "=========================================="
echo "🛑 STOPPING TRAINING JOB"
echo "=========================================="
echo "Job: $JOB_NAME"
echo "Region: $AWS_REGION"
echo ""

# Stop the hyperparameter tuning job
aws sagemaker stop-hyper-parameter-tuning-job \
    --hyper-parameter-tuning-job-name "$JOB_NAME" \
    --region "$AWS_REGION"

echo "✅ Stop command sent"
echo ""
echo "Note: Running jobs will complete their current epoch before stopping."
echo "This may take a few minutes."
echo ""
echo "Check status:"
echo "  python3 $SCRIPT_DIR/start_continuous_training.py --check-status $JOB_NAME"
echo ""
