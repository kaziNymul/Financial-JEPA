#!/bin/bash
# Use SageMaker's pre-built PyTorch container + upload code separately

set -e

echo "=========================================="
echo "📦 PACKAGING CODE FOR SAGEMAKER"
echo "=========================================="

# Get values from Terraform
cd "$(dirname "$0")/../infra"
AWS_REGION=$(terraform output -raw aws_region)
S3_BUCKET=$(terraform output -raw s3_bucket_name)

echo "✓ S3 Bucket: $S3_BUCKET"
echo "✓ Region: $AWS_REGION"

# Package the training code
cd ../../financial-jepa/financial-jepa
echo ""
echo "📦 Packaging training code..."

# Create a source distribution
tar -czf /tmp/sourcedir.tar.gz \
    src/ \
    configs/ \
    artifacts/ \
    requirements.txt \
    kaggle.json

echo "✅ Code packaged: $(du -h /tmp/sourcedir.tar.gz | cut -f1)"

# Upload to S3
echo ""
echo "⬆️  Uploading to S3..."
aws s3 cp /tmp/sourcedir.tar.gz \
    s3://$S3_BUCKET/code/sourcedir.tar.gz \
    --region $AWS_REGION

echo "✅ Code uploaded to S3"

# Clean up
rm /tmp/sourcedir.tar.gz

echo ""
echo "=========================================="
echo "✅ CODE READY"
echo "=========================================="
echo ""
echo "SageMaker will use:"
echo "  • Image: 763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/pytorch-training:2.1.0-gpu-py310"
echo "  • Code: s3://$S3_BUCKET/code/sourcedir.tar.gz"
echo ""
echo "🚀 Ready to start training!"
echo ""
