#!/bin/bash
# Upload training data to S3

set -e

echo "=========================================="
echo "📦 PREPARING DATA FOR AWS SAGEMAKER"
echo "=========================================="

# Get S3 bucket from Terraform
cd "$(dirname "$0")/../infra"
S3_BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

if [ -z "$S3_BUCKET" ]; then
    echo "❌ Error: Could not get S3 bucket from Terraform"
    echo "Please run: terraform apply first"
    exit 1
fi

echo ""
echo "✓ S3 Bucket: $S3_BUCKET"
echo "✓ Region: $AWS_REGION"

# Go to data directory
cd "../../financial-jepa/financial-jepa/data"

echo ""
echo "📤 Uploading processed data shards..."
aws s3 sync processed/data_amex_shards/ s3://$S3_BUCKET/data/processed/ \
    --region $AWS_REGION \
    --exclude "*" \
    --include "*.csv" \
    --quiet

SHARD_COUNT=$(ls processed/data_amex_shards/*.csv 2>/dev/null | wc -l)
echo "✅ Uploaded $SHARD_COUNT shard files"

echo ""
echo "📤 Uploading labels..."
aws s3 cp raw/amex/train_labels.csv s3://$S3_BUCKET/data/labels/ \
    --region $AWS_REGION \
    --quiet
echo "✅ Uploaded train_labels.csv"

echo ""
echo "📤 Uploading config files..."
cd ../
aws s3 sync configs/ s3://$S3_BUCKET/configs/ \
    --region $AWS_REGION \
    --exclude "*" \
    --include "*.yaml" \
    --quiet
echo "✅ Uploaded configuration files"

echo ""
echo "📤 Uploading artifacts..."
aws s3 sync artifacts/ s3://$S3_BUCKET/artifacts/ \
    --region $AWS_REGION \
    --exclude "*" \
    --include "*.json" \
    --exclude "candidates.json" \
    --quiet
echo "✅ Uploaded feature metadata"

echo ""
echo "=========================================="
echo "✅ DATA UPLOAD COMPLETE"
echo "=========================================="
echo ""
echo "📊 Data Summary:"
echo "  • Shards: $SHARD_COUNT CSV files (~16 GB)"
echo "  • Labels: 458,913 customers"
echo "  • Features: 186 features"
echo "  • Location: s3://$S3_BUCKET/data/"
echo ""
echo "🚀 Ready to start training!"
echo ""
