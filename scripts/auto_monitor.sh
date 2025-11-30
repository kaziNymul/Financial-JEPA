#!/bin/bash

# Auto-monitor training job and stop if stuck for >30 minutes
# Usage: ./auto_monitor.sh [job-name]

set -e

REGION="us-east-1"
JOB_NAME="${1:-$(cat current_training_job.txt 2>/dev/null || echo '')}"

if [ -z "$JOB_NAME" ]; then
    echo "❌ No job name provided and current_training_job.txt not found"
    echo "Usage: ./auto_monitor.sh <job-name>"
    exit 1
fi

echo "========================================"
echo "🔍 AUTO-MONITORING: $JOB_NAME"
echo "========================================"
echo "Will stop job if stuck for >30 minutes"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Track last progress timestamp
LAST_PROGRESS_TIME=$(date +%s)
LAST_LOG_LINE=""
CHECK_INTERVAL=60  # Check every 60 seconds
STUCK_THRESHOLD=1800  # 30 minutes = 1800 seconds

while true; do
    echo "[$(date +'%H:%M:%S')] Checking status..."
    
    # Get job status
    JOB_STATUS=$(aws sagemaker describe-hyper-parameter-tuning-job \
        --hyper-parameter-tuning-job-name "$JOB_NAME" \
        --region "$REGION" \
        --query 'HyperParameterTuningJobStatus' \
        --output text 2>/dev/null || echo "NOTFOUND")
    
    if [ "$JOB_STATUS" == "NOTFOUND" ]; then
        echo "❌ Job not found or already completed"
        exit 1
    fi
    
    if [ "$JOB_STATUS" == "Completed" ] || [ "$JOB_STATUS" == "Stopped" ] || [ "$JOB_STATUS" == "Failed" ]; then
        echo "✅ Job finished with status: $JOB_STATUS"
        exit 0
    fi
    
    # Get training job count
    TRAINING_JOBS=$(aws sagemaker describe-hyper-parameter-tuning-job \
        --hyper-parameter-tuning-job-name "$JOB_NAME" \
        --region "$REGION" \
        --query 'TrainingJobStatusCounters' \
        --output json 2>/dev/null)
    
    COMPLETED=$(echo "$TRAINING_JOBS" | jq -r '.Completed // 0')
    IN_PROGRESS=$(echo "$TRAINING_JOBS" | jq -r '.InProgress // 0')
    FAILED=$(echo "$TRAINING_JOBS" | jq -r '.NonRetryableError // 0')
    RETRYABLE=$(echo "$TRAINING_JOBS" | jq -r '.RetryableError // 0')
    STOPPED=$(echo "$TRAINING_JOBS" | jq -r '.Stopped // 0')
    
    echo "  Status: Completed=$COMPLETED InProgress=$IN_PROGRESS Failed=$FAILED Retryable=$RETRYABLE Stopped=$STOPPED"
    
    # Get latest log from any running training job
    RUNNING_JOBS=$(aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job \
        --hyper-parameter-tuning-job-name "$JOB_NAME" \
        --region "$REGION" \
        --status-equals InProgress \
        --query 'TrainingJobSummaries[0].TrainingJobName' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$RUNNING_JOBS" ] && [ "$RUNNING_JOBS" != "None" ]; then
        # Get log stream
        LOG_STREAM=$(aws sagemaker describe-training-job \
            --training-job-name "$RUNNING_JOBS" \
            --region "$REGION" \
            --query 'AlgorithmSpecification.MetricDefinitions[0]' \
            --output text 2>/dev/null || echo "")
        
        # Try to get latest logs
        LATEST_LOGS=$(aws logs tail "/aws/sagemaker/TrainingJobs" \
            --log-stream-name-prefix "$RUNNING_JOBS/algo-1" \
            --since 2m \
            --format short \
            --region "$REGION" 2>/dev/null | tail -3 || echo "")
        
        if [ -n "$LATEST_LOGS" ]; then
            # Check if logs changed
            if [ "$LATEST_LOGS" != "$LAST_LOG_LINE" ]; then
                echo "  📊 Latest: $(echo "$LATEST_LOGS" | tail -1 | cut -c1-100)"
                LAST_LOG_LINE="$LATEST_LOGS"
                LAST_PROGRESS_TIME=$(date +%s)
            else
                # Calculate stuck time
                CURRENT_TIME=$(date +%s)
                STUCK_TIME=$((CURRENT_TIME - LAST_PROGRESS_TIME))
                
                echo "  ⚠️  No new logs for $((STUCK_TIME / 60)) minutes"
                
                if [ $STUCK_TIME -gt $STUCK_THRESHOLD ]; then
                    echo ""
                    echo "🛑 JOB STUCK FOR >30 MINUTES!"
                    echo "Last progress: $(date -d "@$LAST_PROGRESS_TIME" +'%H:%M:%S')"
                    echo "Stopping job..."
                    
                    aws sagemaker stop-hyper-parameter-tuning-job \
                        --hyper-parameter-tuning-job-name "$JOB_NAME" \
                        --region "$REGION"
                    
                    echo "✅ Stop command sent"
                    exit 2
                fi
            fi
        fi
    else
        echo "  ℹ️  No running jobs yet (waiting to start...)"
        # Reset timer if no jobs running yet
        LAST_PROGRESS_TIME=$(date +%s)
    fi
    
    echo ""
    sleep $CHECK_INTERVAL
done
