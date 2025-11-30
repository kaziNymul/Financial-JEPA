#!/bin/bash

# Wait for current 4 jobs to finish, then stop the tuning job
# This prevents the next batch from starting

REGION="us-east-1"
JOB_NAME="jepa-20251130-041149"

echo "========================================"
echo "⏳ WAITING FOR CURRENT BATCH TO FINISH"
echo "========================================"
echo "Job: $JOB_NAME"
echo "Will stop after 4 InProgress jobs complete"
echo ""

while true; do
    # Get status
    STATUS=$(aws sagemaker describe-hyper-parameter-tuning-job \
        --hyper-parameter-tuning-job-name "$JOB_NAME" \
        --region "$REGION" \
        --query 'TrainingJobStatusCounters' \
        --output json)
    
    IN_PROGRESS=$(echo "$STATUS" | jq -r '.InProgress // 0')
    COMPLETED=$(echo "$STATUS" | jq -r '.Completed // 0')
    STOPPED=$(echo "$STATUS" | jq -r '.Stopped // 0')
    
    echo "[$(date +'%H:%M:%S')] InProgress: $IN_PROGRESS, Completed: $COMPLETED, Stopped: $STOPPED"
    
    # Stop when no jobs are in progress (current batch finished)
    if [ "$IN_PROGRESS" -eq 0 ]; then
        echo ""
        echo "✅ Current batch finished!"
        echo "🛑 Stopping tuning job to prevent next batch..."
        
        aws sagemaker stop-hyper-parameter-tuning-job \
            --hyper-parameter-tuning-job-name "$JOB_NAME" \
            --region "$REGION"
        
        echo ""
        echo "✅ Training stopped!"
        echo ""
        echo "Final Status:"
        echo "  Completed: $COMPLETED"
        echo "  Stopped: $STOPPED"
        echo ""
        echo "Models saved in: s3://financial-jepa-data-057149785966/models/$JOB_NAME/"
        exit 0
    fi
    
    sleep 60
done
