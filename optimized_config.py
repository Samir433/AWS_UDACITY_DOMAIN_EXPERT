# Optimized Configuration for LLaMA 2 Fine-tuning Project

import os
from typing import Dict, Any

# Model Configuration
MODEL_CONFIG = {
    "model_id": "meta-textgeneration-llama-2-7b",
    "model_version": "2.*",
}

# Optimized Training Configuration
TRAINING_CONFIG = {
    # Instance optimization - reduced from ml.g5.2xlarge
    "instance_type": "ml.g5.xlarge",  # 40% cost reduction
    "use_spot_instances": True,  # 50-90% additional cost reduction
    "max_wait": 7200,  # 2 hours max wait for spot instances
    "max_run": 3600,   # 1 hour max training time
    
    # Training hyperparameters
    "hyperparameters": {
        "instruction_tuned": "False",
        "epoch": "8",  # Increased max epochs
        "learning_rate": "2e-5",  # Optimized learning rate
        "per_device_train_batch_size": "2",
        "gradient_checkpointing": "True",  # Memory optimization
        "early_stopping_patience": "2",  # Early stopping
        "evaluation_strategy": "steps",
        "eval_steps": "100",
        "save_strategy": "steps",
        "save_steps": "100",
        "logging_steps": "50",
        "warmup_steps": "100",
    }
}

# Optimized Inference Configuration
INFERENCE_CONFIG = {
    # Right-sized instance for inference
    "instance_type": "ml.g5.xlarge",  # Reduced from g5.2xlarge
    "initial_instance_count": 1,
    "serverless": True,  # Enable serverless for cost optimization
    "serverless_config": {
        "memory_size_in_mb": 6144,  # 6GB memory
        "max_concurrency": 1
    }
}

# Dataset Configuration
DATASETS = {
    "finance": "s3://genaiwithawsproject2024/training-datasets/finance",
    "medical": "s3://genaiwithawsproject2024/training-datasets/medical",
    "it": "s3://genaiwithawsproject2024/training-datasets/it"
}

# Test Prompts by Domain
TEST_PROMPTS = {
    "finance": [
        "The investment tests performed indicate",
        "the relative volume for the long out of the money options, indicates",
        "The results for the short in the money options",
        "The results are encouraging for aggressive investors"
    ],
    "medical": [
        "Myeloid neoplasms and acute leukemias derive from",
        "Genomic characterization is essential for",
        "Certain germline disorders may be associated with",
        "In contrast to targeted approaches, genome-wide sequencing"
    ],
    "it": [
        "Traditional approaches to data management such as",
        "A second important aspect of ubiquitous computing environments is",
        "because ubiquitous computing is intended to",
        "outline the key aspects of ubiquitous computing from a data management perspective."
    ]
}

# Prediction Parameters
PREDICTION_PARAMS = {
    "max_new_tokens": 64,
    "top_p": 0.9,
    "temperature": 0.6,
    "return_full_text": False,
}

# Cost Monitoring Configuration
COST_MONITORING = {
    "daily_budget_limit": 50.0,  # $50 daily limit
    "alarm_name": "SageMaker-Daily-Cost-Alert",
    "sns_topic_arn": "arn:aws:sns:us-west-2:087562234367:sagemaker-cost-alerts"  # Update with actual ARN
}

# Hyperparameter Tuning Configuration
HYPERPARAMETER_TUNING = {
    "enabled": False,  # Set to True to enable tuning
    "max_jobs": 10,
    "max_parallel_jobs": 2,
    "objective_metric": "eval_loss",
    "ranges": {
        "learning_rate": (1e-5, 1e-3),
        "per_device_train_batch_size": (1, 4),
        "epochs": (3, 8),
    }
}

# Caching Configuration
CACHE_CONFIG = {
    "enabled": True,
    "cache_dir": "cache",
    "max_cache_size_gb": 5.0
}

def get_domain_config(domain: str) -> Dict[str, Any]:
    """Get configuration for a specific domain"""
    if domain not in DATASETS:
        raise ValueError(f"Unknown domain: {domain}. Available domains: {list(DATASETS.keys())}")
    
    return {
        "dataset": DATASETS[domain],
        "test_prompts": TEST_PROMPTS[domain],
        "domain_name": domain
    }

def get_optimized_training_config(domain: str, enable_spot: bool = True) -> Dict[str, Any]:
    """Get optimized training configuration for a domain"""
    config = TRAINING_CONFIG.copy()
    config["training_data"] = DATASETS[domain]
    
    if not enable_spot:
        config.pop("use_spot_instances", None)
        config.pop("max_wait", None)
        config.pop("max_run", None)
    
    return config

def get_inference_config(use_serverless: bool = True) -> Dict[str, Any]:
    """Get optimized inference configuration"""
    config = INFERENCE_CONFIG.copy()
    
    if not use_serverless:
        config.pop("serverless", None)
        config.pop("serverless_config", None)
    
    return config