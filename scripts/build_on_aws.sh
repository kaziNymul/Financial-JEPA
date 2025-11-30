#!/bin/bash
# Build Docker image on AWS EC2 and push to ECR

set -e

echo "=========================================="
echo "🚀 BUILDING ON AWS"
echo "=========================================="

# Get values from Terraform
cd "$(dirname "$0")/../infra"
ECR_URL=$(terraform output -raw ecr_repository_url)
AWS_REGION=$(terraform output -raw aws_region)
ACCOUNT_ID=$(terraform output -raw account_id)

echo "✓ ECR: $ECR_URL"
echo "✓ Region: $AWS_REGION"

# Create a simple build script
cat > /tmp/aws_build.sh <<'BUILDSCRIPT'
#!/bin/bash
set -e

# Update system
sudo yum update -y
sudo yum install -y docker git

# Start Docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Login to ECR
aws ecr get-login-password --region REGION | sudo docker login --username AWS --password-stdin ECR_URL

# Clone or copy code (we'll use SageMaker's PyTorch container as base)
cd /tmp

# Create Dockerfile
cat > Dockerfile <<'EOF'
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310

# Install additional dependencies
RUN pip install --no-cache-dir \
    pandas>=2.2 \
    pyarrow>=16.1 \
    scikit-learn>=1.5 \
    tqdm>=4.66 \
    pydantic>=2.7 \
    sagemaker-training \
    boto3

WORKDIR /opt/ml/code

ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
EOF

# Build and push
sudo docker build -t financial-jepa-training .
sudo docker tag financial-jepa-training:latest ECR_URL:latest
sudo docker push ECR_URL:latest

echo "✅ Image built and pushed successfully"
BUILDSCRIPT

# Replace placeholders
sed -i "s|REGION|$AWS_REGION|g" /tmp/aws_build.sh
sed -i "s|ECR_URL|$ECR_URL|g" /tmp/aws_build.sh

# Launch EC2 instance with the build script
echo ""
echo "📦 Creating EC2 build instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region $AWS_REGION \
    --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2 \
    --instance-type t3.medium \
    --iam-instance-profile Name=financial-jepa-sagemaker-execution-role \
    --user-data file:///tmp/aws_build.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=docker-builder}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✓ Instance launched: $INSTANCE_ID"
echo ""
echo "⏳ Building Docker image on AWS..."
echo "   This will take 5-10 minutes"
echo ""
echo "📊 Monitor progress:"
echo "   aws ec2 get-console-output --instance-id $INSTANCE_ID --region $AWS_REGION"
echo ""

# Wait for instance to complete
echo "Waiting for build to complete..."
while true; do
    STATE=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $AWS_REGION \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text)
    
    echo -n "."
    
    if [ "$STATE" != "running" ] && [ "$STATE" != "pending" ]; then
        break
    fi
    
    sleep 30
done

echo ""
echo ""

# Get logs
echo "📋 Build logs:"
aws ec2 get-console-output \
    --instance-id $INSTANCE_ID \
    --region $AWS_REGION \
    --output text | tail -50

# Terminate instance
echo ""
echo "🧹 Cleaning up instance..."
aws ec2 terminate-instances \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION > /dev/null

echo ""
echo "=========================================="
echo "✅ BUILD COMPLETE"
echo "=========================================="
echo ""
echo "Verify image:"
echo "  aws ecr describe-images --repository-name financial-jepa-training --region $AWS_REGION"
echo ""
