terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use remote state (recommended for production)
  # backend "s3" {
  #   bucket         = "terraform-state-financial-jepa"
  #   key            = "infra/terraform.tfstate"
  #   region         = "eu-north-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      ManagedBy   = "Terraform"
      Environment = var.environment
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ========================
# S3 Buckets
# ========================

# Main data bucket with encryption
resource "aws_s3_bucket" "jepa_data" {
  bucket        = "${var.s3_bucket_prefix}-${data.aws_caller_identity.current.account_id}"
  force_destroy = var.environment != "prod" # Prevent accidental deletion in prod
}

resource "aws_s3_bucket_versioning" "jepa_data" {
  bucket = aws_s3_bucket.jepa_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "jepa_data" {
  bucket = aws_s3_bucket.jepa_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "jepa_data" {
  bucket                  = aws_s3_bucket.jepa_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy for cost optimization
resource "aws_s3_bucket_lifecycle_configuration" "jepa_data" {
  bucket = aws_s3_bucket.jepa_data.id

  rule {
    id     = "archive-old-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365 # Delete checkpoints older than 1 year
    }
  }

  rule {
    id     = "delete-incomplete-uploads"
    status = "Enabled"

    filter {
      prefix = ""
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# ========================
# ECR Repository for Docker Images
# ========================

resource "aws_ecr_repository" "jepa_training" {
  name                 = "${var.project_name}-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "jepa_training" {
  repository = aws_ecr_repository.jepa_training.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ========================
# IAM Roles
# ========================

# SageMaker Execution Role
resource "aws_iam_role" "sagemaker_execution" {
  name = "${var.project_name}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_policy" {
  name = "${var.project_name}-sagemaker-policy"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 access
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.jepa_data.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.jepa_data.arn}/*"
        ]
      },
      # ECR access
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      # CloudWatch Logs
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      },
      # CloudWatch Metrics
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      },
      # SageMaker
      {
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeTrainingJob",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DescribeModel"
        ]
        Resource = "*"
      }
    ]
  })
}

# ========================
# Security Group
# ========================

resource "aws_security_group" "sagemaker" {
  name        = "${var.project_name}-sagemaker-sg"
  description = "Security group for SageMaker resources"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name = "${var.project_name}-sagemaker-sg"
  }
}

# ========================
# SageMaker Notebook Instance
# ========================

# Lifecycle configuration for auto-stop
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "auto_stop" {
  name = "${var.project_name}-auto-stop"

  on_start = base64encode(<<-EOF
    #!/bin/bash
    set -e
    
    # Install additional packages
    sudo -u ec2-user -i <<'EOC'
    source /home/ec2-user/anaconda3/bin/activate pytorch_p310
    pip install --upgrade pip
    pip install tensorboard umap-learn polars
    EOC
    
    echo "Notebook instance started successfully"
  EOF
  )

  on_create = base64encode(<<-EOF
    #!/bin/bash
    set -e
    
    # Auto-stop after idle timeout (cost savings)
    echo "Setting up auto-stop for idle timeout..."
    
    cat > /home/ec2-user/SageMaker/auto-stop.py << 'PYTHON'
import time
import boto3

IDLE_TIME = ${var.notebook_idle_timeout_minutes} * 60  # Convert to seconds

def check_idle_time():
    client = boto3.client('sagemaker')
    notebook_name = '${var.project_name}-notebook'
    
    try:
        response = client.describe_notebook_instance(NotebookInstanceName=notebook_name)
        last_modified = response['LastModifiedTime']
        
        idle_seconds = (time.time() - last_modified.timestamp())
        
        if idle_seconds > IDLE_TIME:
            print(f"Notebook idle for {idle_seconds/60:.1f} minutes. Stopping...")
            client.stop_notebook_instance(NotebookInstanceName=notebook_name)
        else:
            print(f"Notebook active. Idle for {idle_seconds/60:.1f} minutes.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_idle_time()
PYTHON
    
    # Set up cron job to check every 5 minutes
    (crontab -l 2>/dev/null; echo "*/5 * * * * /home/ec2-user/anaconda3/bin/python /home/ec2-user/SageMaker/auto-stop.py") | crontab -
    
    echo "Auto-stop configured successfully"
  EOF
  )
}

resource "aws_sagemaker_notebook_instance" "jepa_notebook" {
  count = var.enable_notebook ? 1 : 0

  name                    = "${var.project_name}-notebook"
  instance_type           = var.notebook_instance_type
  role_arn                = aws_iam_role.sagemaker_execution.arn
  subnet_id               = data.aws_subnets.default.ids[0]
  security_groups         = [aws_security_group.sagemaker.id]
  volume_size             = var.notebook_volume_size
  direct_internet_access  = "Enabled"
  lifecycle_config_name   = aws_sagemaker_notebook_instance_lifecycle_configuration.auto_stop.name
  platform_identifier     = "notebook-al2-v2" # Amazon Linux 2

  tags = {
    Name        = "${var.project_name}-notebook"
    Environment = var.environment
  }
}

# ========================
# CloudWatch Log Groups
# ========================

resource "aws_cloudwatch_log_group" "training_jobs" {
  name              = "/aws/sagemaker/TrainingJobs/${var.project_name}"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-training-logs"
  }
}

resource "aws_cloudwatch_log_group" "endpoints" {
  name              = "/aws/sagemaker/Endpoints/${var.project_name}"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-endpoint-logs"
  }
}

# ========================
# SNS Topic for Alerts
# ========================

resource "aws_sns_topic" "alerts" {
  count = var.enable_monitoring ? 1 : 0

  name = "${var.project_name}-alerts"

  tags = {
    Name = "${var.project_name}-alerts"
  }
}

resource "aws_sns_topic_subscription" "email_alerts" {
  count = var.enable_monitoring && var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ========================
# CloudWatch Alarms
# ========================

# Alarm for notebook instance status
resource "aws_cloudwatch_metric_alarm" "notebook_status" {
  count = var.enable_monitoring && var.enable_notebook ? 1 : 0

  alarm_name          = "${var.project_name}-notebook-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "NotebookInstanceStatus"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "Triggers when notebook instance fails"
  alarm_actions       = [aws_sns_topic.alerts[0].arn]

  dimensions = {
    NotebookInstanceName = aws_sagemaker_notebook_instance.jepa_notebook[0].name
  }
}

# ========================
# Outputs
# ========================

output "s3_bucket_name" {
  description = "Name of the S3 bucket for data and models"
  value       = aws_s3_bucket.jepa_data.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.jepa_data.arn
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.jepa_training.repository_url
}

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "notebook_name" {
  description = "Name of the SageMaker notebook instance"
  value       = var.enable_notebook ? aws_sagemaker_notebook_instance.jepa_notebook[0].name : "Not enabled"
}

output "notebook_url" {
  description = "URL to access the notebook instance"
  value       = var.enable_notebook ? "https://console.aws.amazon.com/sagemaker/home?region=${data.aws_region.current.name}#/notebook-instances/${aws_sagemaker_notebook_instance.jepa_notebook[0].name}" : "Not enabled"
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = var.enable_monitoring ? aws_sns_topic.alerts[0].arn : "Not enabled"
}

output "aws_region" {
  description = "AWS region where resources are deployed"
  value       = data.aws_region.current.name
}

output "account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}
