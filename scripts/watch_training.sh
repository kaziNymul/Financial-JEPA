#!/bin/bash
# Watch training progress with auto-refresh

SCRIPT_DIR="$(dirname "$0")"
JOB_NAME="${1:-$(cat $SCRIPT_DIR/../infra/current_training_job.txt 2>/dev/null)}"

if [ -z "$JOB_NAME" ]; then
    echo "Usage: $0 [job-name]"
    exit 1
fi

echo "Monitoring: $JOB_NAME"
echo "Press Ctrl+C to stop (training continues)"
echo ""

while true; do
    clear
    echo "================================================================================"
    echo "Training Progress - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""
    
    # Get overall status
    python3 "$SCRIPT_DIR/start_continuous_training.py" --check-status "$JOB_NAME"
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Individual Jobs:"
    echo "--------------------------------------------------------------------------------"
    
    # Get individual job details
    aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job \
        --hyper-parameter-tuning-job-name "$JOB_NAME" \
        --region us-east-1 \
        --query 'TrainingJobSummaries[*].[TrainingJobName, TrainingJobStatus, FinalHyperParameterTuningJobObjectiveMetric.Value, TunedHyperParameters."learning-rate"]' \
        --output table 2>/dev/null || echo "No jobs yet"
    
    echo ""
    echo "Next refresh in 60 seconds..."
    sleep 60
done
