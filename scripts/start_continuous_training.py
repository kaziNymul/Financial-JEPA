#!/usr/bin/env python3
"""
Start 48-hour continuous training on AWS SageMaker
Runs 60 experiments with Bayesian optimization using all 186 features
"""

import boto3
import json
import os
from datetime import datetime
import argparse

def get_aws_config():
    """Get AWS configuration from Terraform outputs or environment"""
    try:
        import subprocess
        os.chdir('../infra')
        result = subprocess.run(['terraform', 'output', '-json'], 
                              capture_output=True, text=True, check=True)
        outputs = json.loads(result.stdout)
        
        return {
            'ecr_url': outputs['ecr_repository_url']['value'],
            'role_arn': outputs['sagemaker_role_arn']['value'],
            'bucket': outputs['s3_bucket_name']['value'],
            'region': outputs['aws_region']['value']
        }
    except Exception as e:
        print(f"Error reading Terraform outputs: {e}")
        print("Using environment variables instead...")
        return {
            'ecr_url': os.environ.get('ECR_URL'),
            'role_arn': os.environ.get('SAGEMAKER_ROLE_ARN'),
            'bucket': os.environ.get('S3_BUCKET'),
            'region': os.environ.get('AWS_REGION', 'us-east-1')
        }

def create_training_job(config, parallel_jobs=4, max_jobs=60):
    """Create SageMaker Hyperparameter Tuning Job for continuous training"""
    
    sagemaker = boto3.client('sagemaker', region_name=config['region'])
    
    job_name = f"jepa-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    tuning_config = {
        'HyperParameterTuningJobName': job_name,
        'HyperParameterTuningJobConfig': {
            'Strategy': 'Bayesian',
            'HyperParameterTuningJobObjective': {
                'Type': 'Minimize',
                'MetricName': 'validation:loss'
            },
            'ResourceLimits': {
                'MaxNumberOfTrainingJobs': max_jobs,
                'MaxParallelTrainingJobs': parallel_jobs,
            },
            'ParameterRanges': {
                'ContinuousParameterRanges': [
                    {
                        'Name': 'learning-rate',
                        'MinValue': '0.00001',
                        'MaxValue': '0.001',
                        'ScalingType': 'Logarithmic'
                    },
                    {
                        'Name': 'time-mask-prob',
                        'MinValue': '0.10',
                        'MaxValue': '0.25',
                        'ScalingType': 'Linear'
                    },
                    {
                        'Name': 'feature-mask-prob',
                        'MinValue': '0.05',
                        'MaxValue': '0.15',
                        'ScalingType': 'Linear'
                    },
                    {
                        'Name': 'dropout',
                        'MinValue': '0.05',
                        'MaxValue': '0.20',
                        'ScalingType': 'Linear'
                    },
                    {
                        'Name': 'ema-decay',
                        'MinValue': '0.995',
                        'MaxValue': '0.999',
                        'ScalingType': 'Linear'
                    },
                    {
                        'Name': 'weight-decay',
                        'MinValue': '0.000001',
                        'MaxValue': '0.0001',
                        'ScalingType': 'Logarithmic'
                    },
                    {
                        'Name': 'grad-clip',
                        'MinValue': '0.5',
                        'MaxValue': '2.0',
                        'ScalingType': 'Linear'
                    },
                ],
                'IntegerParameterRanges': [
                    {
                        'Name': 'batch-size',
                        'MinValue': '32',
                        'MaxValue': '128',
                        'ScalingType': 'Linear'
                    },
                    {
                        'Name': 'd-model',
                        'MinValue': '128',
                        'MaxValue': '320',
                        'ScalingType': 'Linear'
                    },
                    {
                        'Name': 'n-layers',
                        'MinValue': '1',
                        'MaxValue': '3',
                        'ScalingType': 'Linear'
                    },
                ]
            }
        },
        'TrainingJobDefinition': {
            'StaticHyperParameters': {
                'config': 'configs/jepa_improved.yaml',
                'epochs': '12',
                'mixed-precision': '1',
                'sagemaker-mode': '1',
                'encoder': 'gru',
                'sagemaker_program': 'src/train.py',
                'sagemaker_submit_directory': f"s3://{config['bucket']}/code/sourcedir.tar.gz"
            },
            'AlgorithmSpecification': {
                'TrainingImage': f"763104351884.dkr.ecr.{config['region']}.amazonaws.com/pytorch-training:2.1.0-gpu-py310",
                'TrainingInputMode': 'File',
                'MetricDefinitions': [
                    {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                    {'Name': 'validation:loss', 'Regex': 'validation:loss.*value=([0-9\\.]+)'},
                ]
            },
            'RoleArn': config['role_arn'],
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{config['bucket']}/data/processed/",
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv',
                    'CompressionType': 'None'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f"s3://{config['bucket']}/models/"
            },
            'ResourceConfig': {
                'InstanceType': 'ml.g4dn.xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 120
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 21600  # 6 hours max per job
            },
            'EnableManagedSpotTraining': False,
            'CheckpointConfig': {
                'S3Uri': f"s3://{config['bucket']}/checkpoints/",
                'LocalPath': '/opt/ml/checkpoints'
            }
        },
        'Tags': [
            {'Key': 'Project', 'Value': 'financial-jepa'},
            {'Key': 'Training', 'Value': 'continuous-48h'},
            {'Key': 'Features', 'Value': 'all-186'}
        ]
    }
    
    print("\n" + "="*80)
    print("🚀 STARTING 48-HOUR NON-STOP TRAINING")
    print("="*80)
    print(f"\nJob Name: {job_name}")
    print(f"Strategy: Bayesian Optimization")
    print(f"Parallel Jobs: {parallel_jobs} simultaneous")
    print(f"Total Jobs: {max_jobs} experiments")
    print(f"Features: ALL 186 features")
    print(f"Instance: ml.g4dn.xlarge (GPU, On-Demand)")
    estimated_hours = (max_jobs / parallel_jobs) * 4  # Avg 4 hours per job with early stopping
    estimated_cost = estimated_hours * 0.526 * parallel_jobs
    print(f"Estimated Cost: ~${estimated_cost:.2f}")
    print(f"Estimated Duration: {max_jobs / parallel_jobs * 6:.0f} hours (max)")
    print(f"\nS3 Output: s3://{config['bucket']}/models/")
    print(f"Checkpoints: s3://{config['bucket']}/checkpoints/")
    
    try:
        response = sagemaker.create_hyper_parameter_tuning_job(**tuning_config)
        
        print("\n✅ Training job started successfully!")
        print(f"\nJob ARN: {response['HyperParameterTuningJobArn']}")
        print(f"\n📊 Monitor progress:")
        print(f"   aws sagemaker describe-hyper-parameter-tuning-job \\")
        print(f"     --hyper-parameter-tuning-job-name {job_name}")
        print(f"\n🔍 View in AWS Console:")
        print(f"   https://console.aws.amazon.com/sagemaker/home?region={config['region']}#/hyper-tuning-jobs/{job_name}")
        print("\n⏰ Training will run non-stop for 48 hours!")
        print("   You can safely close this terminal - training continues on AWS")
        print("="*80 + "\n")
        
        return job_name
        
    except Exception as e:
        print(f"\n❌ Error starting training job: {e}")
        raise

def check_status(job_name, region):
    """Check status of training job"""
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sagemaker.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )
        
        status = response['HyperParameterTuningJobStatus']
        counts = response['TrainingJobStatusCounters']
        best = response.get('BestTrainingJob', {})
        
        print("\n" + "="*80)
        print("📊 TRAINING STATUS")
        print("="*80)
        print(f"\nJob: {job_name}")
        print(f"Status: {status}")
        print(f"\nProgress:")
        print(f"  ✅ Completed: {counts['Completed']}")
        print(f"  🔄 InProgress: {counts['InProgress']}")
        print(f"  ⏸️  Stopped: {counts['Stopped']}")
        print(f"  ❌ Failed: {counts.get('Failed', 0)}")
        
        if best:
            print(f"\n🏆 Best Model So Far:")
            print(f"  Job: {best['TrainingJobName']}")
            print(f"  Validation Loss: {best['FinalHyperParameterTuningJobObjectiveMetric']['Value']:.6f}")
            print(f"  Status: {best['TrainingJobStatus']}")
        
        print("="*80 + "\n")
        
        return response
        
    except Exception as e:
        print(f"Error checking status: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Start continuous 48-hour training on AWS SageMaker')
    parser.add_argument('--parallel-jobs', type=int, default=4, help='Number of parallel training jobs (default: 4)')
    parser.add_argument('--max-jobs', type=int, default=60, help='Maximum total training jobs (default: 60)')
    parser.add_argument('--check-status', type=str, help='Check status of existing job')
    args = parser.parse_args()
    
    # Get AWS configuration
    config = get_aws_config()
    
    # Verify configuration
    if not all(config.values()):
        print("❌ Error: Missing AWS configuration")
        print("Please ensure Terraform is deployed or set these environment variables:")
        print("  - ECR_URL")
        print("  - SAGEMAKER_ROLE_ARN")
        print("  - S3_BUCKET")
        print("  - AWS_REGION")
        return 1
    
    if args.check_status:
        check_status(args.check_status, config['region'])
    else:
        job_name = create_training_job(config, args.parallel_jobs, args.max_jobs)
        
        # Save job name for later reference
        with open('current_training_job.txt', 'w') as f:
            f.write(job_name)
        
        print(f"💾 Job name saved to: current_training_job.txt")
    
    return 0

if __name__ == '__main__':
    exit(main())
