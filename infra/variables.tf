# ========================
# Core Variables
# ========================

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-north-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]{1}$", var.aws_region))
    error_message = "AWS region must be in format: us-east-1, eu-west-1, etc."
  }
}

variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "financial-jepa"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# ========================
# S3 Configuration
# ========================

variable "s3_bucket_prefix" {
  description = "Prefix for S3 bucket name (account ID will be appended for uniqueness)"
  type        = string
  default     = "financial-jepa-data"
}

variable "enable_s3_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

# ========================
# Notebook Configuration
# ========================

variable "enable_notebook" {
  description = "Whether to create SageMaker notebook instance"
  type        = bool
  default     = true
}

variable "notebook_instance_type" {
  description = "SageMaker notebook instance type"
  type        = string
  default     = "ml.t3.medium" # Cheaper for development/debugging

  validation {
    condition     = can(regex("^ml\\.", var.notebook_instance_type))
    error_message = "Instance type must start with 'ml.' (e.g., ml.t3.medium, ml.g5.xlarge)."
  }
}

variable "notebook_volume_size" {
  description = "EBS volume size in GB for notebook instance"
  type        = number
  default     = 100

  validation {
    condition     = var.notebook_volume_size >= 5 && var.notebook_volume_size <= 16384
    error_message = "Volume size must be between 5 and 16384 GB."
  }
}

variable "notebook_idle_timeout_minutes" {
  description = "Auto-stop notebook after this many minutes of idle time (cost savings)"
  type        = number
  default     = 120 # 2 hours

  validation {
    condition     = var.notebook_idle_timeout_minutes >= 30
    error_message = "Idle timeout must be at least 30 minutes."
  }
}

# ========================
# Training Configuration
# ========================

variable "enable_spot_instances" {
  description = "Use managed spot instances for training (60-70% cost savings)"
  type        = bool
  default     = true
}

variable "max_training_runtime_hours" {
  description = "Maximum runtime for training jobs in hours"
  type        = number
  default     = 24

  validation {
    condition     = var.max_training_runtime_hours > 0 && var.max_training_runtime_hours <= 72
    error_message = "Training runtime must be between 1 and 72 hours."
  }
}

variable "training_instance_type" {
  description = "Instance type for SageMaker training jobs"
  type        = string
  default     = "ml.g5.2xlarge" # Optimal for Financial-JEPA (186 features, batch_size=64)

  validation {
    condition     = can(regex("^ml\\.", var.training_instance_type))
    error_message = "Instance type must start with 'ml.'."
  }
}

variable "training_instance_count" {
  description = "Number of instances for distributed training"
  type        = number
  default     = 1

  validation {
    condition     = var.training_instance_count > 0 && var.training_instance_count <= 20
    error_message = "Instance count must be between 1 and 20."
  }
}

# ========================
# Endpoint Configuration
# ========================

variable "enable_endpoint" {
  description = "Whether to create SageMaker endpoint for inference"
  type        = bool
  default     = false # Disabled by default to save costs
}

variable "endpoint_instance_type" {
  description = "Instance type for SageMaker inference endpoint"
  type        = string
  default     = "ml.m5.large"

  validation {
    condition     = can(regex("^ml\\.", var.endpoint_instance_type))
    error_message = "Instance type must start with 'ml.'."
  }
}

variable "endpoint_min_capacity" {
  description = "Minimum number of endpoint instances for auto-scaling"
  type        = number
  default     = 1

  validation {
    condition     = var.endpoint_min_capacity > 0
    error_message = "Minimum capacity must be at least 1."
  }
}

variable "endpoint_max_capacity" {
  description = "Maximum number of endpoint instances for auto-scaling"
  type        = number
  default     = 3

  validation {
    condition     = var.endpoint_max_capacity >= var.endpoint_min_capacity
    error_message = "Maximum capacity must be >= minimum capacity."
  }
}

variable "endpoint_target_invocations_per_instance" {
  description = "Target number of invocations per instance for auto-scaling"
  type        = number
  default     = 1000
}

# ========================
# Monitoring Configuration
# ========================

variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring and alarms"
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email address for CloudWatch alarms (leave empty to skip email alerts)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch Logs retention period."
  }
}

# ========================
# Cost Optimization
# ========================

variable "enable_s3_lifecycle_policies" {
  description = "Enable S3 lifecycle policies to reduce storage costs"
  type        = bool
  default     = true
}

variable "checkpoint_archive_days" {
  description = "Days before archiving old checkpoints to cheaper storage"
  type        = number
  default     = 30

  validation {
    condition     = var.checkpoint_archive_days >= 0
    error_message = "Archive days must be non-negative."
  }
}

variable "checkpoint_deletion_days" {
  description = "Days before deleting old checkpoints (0 = never delete)"
  type        = number
  default     = 365

  validation {
    condition     = var.checkpoint_deletion_days >= 0
    error_message = "Deletion days must be non-negative."
  }
}

# ========================
# Security Configuration
# ========================

variable "enable_kms_encryption" {
  description = "Use KMS for S3 bucket encryption instead of AES256"
  type        = bool
  default     = false # KMS adds cost
}

variable "kms_key_id" {
  description = "KMS key ID for encryption (if enable_kms_encryption is true)"
  type        = string
  default     = ""
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access resources (empty = all)"
  type        = list(string)
  default     = []
}

# ========================
# Tags
# ========================

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# ========================
# Feature Flags
# ========================

variable "enable_distributed_training" {
  description = "Enable distributed training support"
  type        = bool
  default     = false
}

variable "enable_hyperparameter_tuning" {
  description = "Enable SageMaker hyperparameter tuning jobs"
  type        = bool
  default     = false
}

variable "enable_model_registry" {
  description = "Enable SageMaker Model Registry for model versioning"
  type        = bool
  default     = false
}

variable "enable_vpc_endpoints" {
  description = "Create VPC endpoints for S3 and SageMaker (more secure, lower cost)"
  type        = bool
  default     = false
}

# ========================
# Docker Configuration
# ========================

variable "docker_image_tag" {
  description = "Tag for Docker images in ECR"
  type        = string
  default     = "latest"
}

variable "ecr_image_scan_on_push" {
  description = "Enable vulnerability scanning for Docker images"
  type        = bool
  default     = true
}

# ========================
# Data Configuration
# ========================

variable "data_input_s3_path" {
  description = "S3 path for input training data (optional)"
  type        = string
  default     = ""
}

variable "model_output_s3_path" {
  description = "S3 path for model outputs (optional, defaults to bucket/models/)"
  type        = string
  default     = ""
}
