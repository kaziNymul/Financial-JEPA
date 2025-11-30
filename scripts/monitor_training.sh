#!/bin/bash
# Monitor training progress in real-time

set -e

# Get job name
SCRIPT_DIR="$(dirname "$0")"
if [ -f "$SCRIPT_DIR/../infra/current_training_job.txt" ]; then
    JOB_NAME=$(cat "$SCRIPT_DIR/../infra/current_training_job.txt")
elif [ -f "$SCRIPT_DIR/current_training_job.txt" ]; then
    JOB_NAME=$(cat "$SCRIPT_DIR/current_training_job.txt")
else
    echo "❌ No training job found"
    echo "Usage: $0 [job-name]"
    echo "Or run after starting training to use saved job name"
    exit 1
fi

if [ ! -z "$1" ]; then
    JOB_NAME="$1"
fi

cd "$(dirname "$0")/../infra"
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

echo "=========================================="
echo "📊 MONITORING TRAINING JOB"
echo "=========================================="
echo "Job: $JOB_NAME"
echo "Region: $AWS_REGION"
echo ""
echo "Press Ctrl+C to stop monitoring (training continues)"
echo "=========================================="
echo ""

# Monitor loop
while true; do
    clear
    echo "🔄 Refreshing... ($(date))"
    echo ""
    
    # Get status
    python3 ../scripts/start_continuous_training.py --check-status "$JOB_NAME"
    
    # Sleep for 60 seconds
    echo "Next update in 60 seconds..."
    sleep 60
done
