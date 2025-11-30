# ========================
# SageMaker Training Job Module
# ========================
# This module creates training job configuration
# Requires: ECR image built and pushed

# ========================
# Training Job Configuration (Template)
# ========================
# Note: Training jobs are typically created dynamically via SDK/CLI
# This provides the configuration template

locals {
  training_job_config = {
    training_image       = "${aws_ecr_repository.jepa_training.repository_url}:${var.docker_image_tag}"
    role_arn            = aws_iam_role.sagemaker_execution.arn
    instance_type       = var.training_instance_type
    instance_count      = var.training_instance_count
    volume_size_gb      = 50
    max_runtime_seconds = var.max_training_runtime_hours * 3600
    use_spot_instances  = var.enable_spot_instances
    
    # Hyperparameters (optimized for ml.g5.2xlarge + GRU)
    hyperparameters = {
      "config"           = "configs/jepa_improved.yaml"
      "epochs"           = "12"
      "batch-size"       = "64" # Optimal for ml.g5.2xlarge (24GB VRAM)
      "learning-rate"    = "5e-4"
      "d-model"          = "192" # GRU encoder dimension
      "encoder"          = "gru" # Recommended: faster and effective for time-series
      "time-mask-prob"   = "0.15" # JEPA-style masking
      "mixed-precision"  = "1" # Enable AMP for 30% speedup
      "sagemaker-mode"   = "1"
    }
    
    # Input data channels
    input_data_config = [
      {
        channel_name     = "training"
        data_source = {
          s3_data_source = {
            s3_data_type             = "S3Prefix"
            s3_uri                   = "s3://${aws_s3_bucket.jepa_data.bucket}/data/processed/"
            s3_data_distribution_type = "FullyReplicated"
          }
        }
        compression_type = "None"
        content_type     = "application/x-parquet" # or csv
      }
    ]
    
    # Output configuration
    output_data_config = {
      s3_output_path = "s3://${aws_s3_bucket.jepa_data.bucket}/models/"
    }
    
    # Checkpointing
    checkpoint_config = {
      s3_uri                = "s3://${aws_s3_bucket.jepa_data.bucket}/checkpoints/"
      local_path            = "/opt/ml/checkpoints"
    }
    
    # Metric definitions for hyperparameter tuning
    metric_definitions = [
      {
        name  = "train:loss"
        regex = "train/loss: ([0-9\\.]+)"
      },
      {
        name  = "validation:loss"
        regex = "val/loss: ([0-9\\.]+)"
      },
      {
        name  = "repr:collapse_fraction"
        regex = "repr/collapse_fraction: ([0-9\\.]+)"
      },
      {
        name  = "repr:effective_rank"
        regex = "repr/effective_rank: ([0-9\\.]+)"
      }
    ]
  }
}

# Export training configuration for use in Python/CLI
resource "local_file" "training_config" {
  content = jsonencode({
    TrainingJobName     = "${var.project_name}-training-job"
    RoleArn            = aws_iam_role.sagemaker_execution.arn
    AlgorithmSpecification = {
      TrainingImage     = local.training_job_config.training_image
      TrainingInputMode = "File"
      MetricDefinitions = local.training_job_config.metric_definitions
    }
    ResourceConfig = {
      InstanceType       = local.training_job_config.instance_type
      InstanceCount      = local.training_job_config.instance_count
      VolumeSizeInGB     = local.training_job_config.volume_size_gb
    }
    StoppingCondition = {
      MaxRuntimeInSeconds = local.training_job_config.max_runtime_seconds
    }
    EnableManagedSpotTraining = local.training_job_config.use_spot_instances
    MaxWaitTimeInSeconds      = local.training_job_config.max_runtime_seconds * 2
    HyperParameters           = local.training_job_config.hyperparameters
    InputDataConfig           = local.training_job_config.input_data_config
    OutputDataConfig          = local.training_job_config.output_data_config
    CheckpointConfig          = local.training_job_config.checkpoint_config
    VpcConfig = {
      SecurityGroupIds = [aws_security_group.sagemaker.id]
      Subnets         = data.aws_subnets.default.ids
    }
    Tags = [
      {
        Key   = "Project"
        Value = var.project_name
      },
      {
        Key   = "Environment"
        Value = var.environment
      }
    ]
  })
  
  filename = "${path.module}/training_job_config.json"
}

# ========================
# Hyperparameter Tuning Job (Optional)
# ========================

locals {
  tuning_config = var.enable_hyperparameter_tuning ? {
    strategy = "Bayesian" # or "Random"
    objective = {
      type        = "Minimize"
      metric_name = "validation:loss"
    }
    parameter_ranges = {
      continuous_ranges = [
        {
          name        = "learning-rate"
          min_value   = "1e-5"
          max_value   = "1e-3"
          scaling_type = "Logarithmic"
        },
        {
          name        = "time-mask-prob"
          min_value   = "0.05"
          max_value   = "0.30"
          scaling_type = "Linear"
        }
      ]
      integer_ranges = [
        {
          name        = "batch-size"
          min_value   = "32"
          max_value   = "128" # ml.g5.2xlarge can handle up to 128
          scaling_type = "Linear"
        },
        {
          name        = "d-model"
          min_value   = "128"
          max_value   = "256" # Keep reasonable for 186 features
          scaling_type = "Linear"
        }
      ]
      categorical_ranges = [
        {
          name   = "encoder"
          values = ["gru"] # Focus on GRU (best for time-series)
        }
      ]
    }
    max_jobs           = 20
    max_parallel_jobs  = 2
  } : null
}

# ========================
# Outputs
# ========================

output "training_image_uri" {
  description = "Docker image URI for training jobs"
  value       = local.training_job_config.training_image
}

output "training_role_arn" {
  description = "IAM role ARN for training jobs"
  value       = local.training_job_config.role_arn
}

output "training_data_s3_path" {
  description = "S3 path for training data"
  value       = "s3://${aws_s3_bucket.jepa_data.bucket}/data/processed/"
}

output "model_output_s3_path" {
  description = "S3 path for model outputs"
  value       = "s3://${aws_s3_bucket.jepa_data.bucket}/models/"
}

output "checkpoint_s3_path" {
  description = "S3 path for checkpoints"
  value       = "s3://${aws_s3_bucket.jepa_data.bucket}/checkpoints/"
}

output "training_config_file" {
  description = "Path to training job configuration file"
  value       = local_file.training_config.filename
}
