#!/bin/bash
# Master script to deploy everything and start training

set -e

echo "=========================================="
echo "🚀 FINANCIAL-JEPA DEPLOYMENT"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Deploy AWS infrastructure"
echo "2. Upload training data"
echo "3. Build and push Docker image"
echo "4. Start 48-hour training"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

cd "$(dirname "$0")"

# Step 1: Deploy infrastructure
echo ""
echo "=========================================="
echo "STEP 1/4: Deploying Infrastructure"
echo "=========================================="
cd ../infra
terraform init
terraform apply -auto-approve

# Step 2: Upload data
echo ""
echo "=========================================="
echo "STEP 2/4: Uploading Data"
echo "=========================================="
cd ../scripts
./prepare_data.sh

# Step 3: Build Docker
echo ""
echo "=========================================="
echo "STEP 3/4: Building Docker Image"
echo "=========================================="
./build_and_push_docker.sh

# Step 4: Start training
echo ""
echo "=========================================="
echo "STEP 4/4: Starting Training"
echo "=========================================="
python3 start_continuous_training.py

echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Training is now running for 48 hours!"
echo ""
echo "Monitor progress:"
echo "  ./monitor_training.sh"
echo ""
echo "Or check AWS Console:"
echo "  https://console.aws.amazon.com/sagemaker/"
echo ""
