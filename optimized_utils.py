# Optimized Utilities for LLaMA 2 Fine-tuning Project

import os
import time
import pickle
import logging
import boto3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor training and inference performance"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.start_time = None
        self.metrics = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation"""
        self.start_time = time.time()
        logger.info(f"Starting {operation}")
    
    def end_timing(self, operation: str):
        """End timing and log the operation"""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics[operation] = duration
            logger.info(f"{operation} completed in {duration:.2f} seconds")
            return duration
        return None
    
    def log_custom_metric(self, metric_name: str, value: float, unit: str = 'None'):
        """Log custom metrics to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='SageMaker/Optimization',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
            logger.info(f"Logged metric {metric_name}: {value} {unit}")
        except Exception as e:
            logger.error(f"Failed to log metric {metric_name}: {e}")

class ResourceManager:
    """Manage SageMaker resources with automatic cleanup"""
    
    def __init__(self):
        self.active_predictors = []
        self.active_endpoints = []
    
    @contextmanager
    def managed_predictor(self, predictor):
        """Context manager for automatic predictor cleanup"""
        self.active_predictors.append(predictor)
        try:
            yield predictor
        finally:
            self.cleanup_predictor(predictor)
    
    def cleanup_predictor(self, predictor):
        """Safely cleanup a predictor"""
        try:
            predictor.delete_endpoint()
            predictor.delete_model()
            logger.info("Successfully cleaned up predictor")
            if predictor in self.active_predictors:
                self.active_predictors.remove(predictor)
        except Exception as e:
            logger.error(f"Failed to cleanup predictor: {e}")
    
    def cleanup_all(self):
        """Cleanup all active resources"""
        for predictor in self.active_predictors.copy():
            self.cleanup_predictor(predictor)

class ModelCache:
    """Cache model responses to avoid redundant predictions"""
    
    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 5.0):
        self.cache_dir = cache_dir
        self.max_size_gb = max_size_gb
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_file(self, model_name: str, input_text: str) -> str:
        """Generate cache file path"""
        cache_key = hash(f"{model_name}_{input_text}")
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get_cached_response(self, model_name: str, input_text: str) -> Optional[Any]:
        """Get cached response if available"""
        cache_file = self._get_cache_file(model_name, input_text)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    logger.info(f"Cache hit for input: {input_text[:50]}...")
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load cached response: {e}")
        
        return None
    
    def cache_response(self, model_name: str, input_text: str, response: Any):
        """Cache a model response"""
        cache_file = self._get_cache_file(model_name, input_text)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
            logger.info(f"Cached response for input: {input_text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    def get_cache_size(self) -> float:
        """Get current cache size in GB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024**3)  # Convert to GB
    
    def cleanup_cache(self):
        """Clean up cache if it exceeds max size"""
        current_size = self.get_cache_size()
        if current_size > self.max_size_gb:
            logger.warning(f"Cache size ({current_size:.2f}GB) exceeds limit ({self.max_size_gb}GB)")
            # Remove oldest files first
            cache_files = []
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    cache_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time
            cache_files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            for filepath, _ in cache_files:
                os.remove(filepath)
                if self.get_cache_size() <= self.max_size_gb:
                    break
            
            logger.info(f"Cache cleaned up. New size: {self.get_cache_size():.2f}GB")

def setup_cost_monitoring(daily_limit: float = 50.0, sns_topic_arn: Optional[str] = None):
    """Set up CloudWatch cost monitoring and alerts"""
    cloudwatch = boto3.client('cloudwatch')
    
    try:
        alarm_actions = [sns_topic_arn] if sns_topic_arn else []
        
        cloudwatch.put_metric_alarm(
            AlarmName='SageMaker-Daily-Cost-Alert',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName='EstimatedCharges',
            Namespace='AWS/Billing',
            Period=86400,  # Daily
            Statistic='Maximum',
            Threshold=daily_limit,
            ActionsEnabled=True,
            AlarmActions=alarm_actions,
            AlarmDescription=f'Alert when daily SageMaker costs exceed ${daily_limit}'
        )
        logger.info(f"Cost monitoring alarm set up with ${daily_limit} daily limit")
    except Exception as e:
        logger.error(f"Failed to set up cost monitoring: {e}")

def batch_predict(predictor, inputs_list: List[str], batch_size: int = 4, 
                 prediction_params: Optional[Dict[str, Any]] = None, cache: Optional[ModelCache] = None,
                 model_name: str = "llama2") -> List[Any]:
    """Optimized batch prediction with caching"""
    if prediction_params is None:
        prediction_params = {
            "max_new_tokens": 64,
            "top_p": 0.9,
            "temperature": 0.6,
            "return_full_text": False,
        }
    
    results = []
    cache_hits = 0
    
    for i in range(0, len(inputs_list), batch_size):
        batch_inputs = inputs_list[i:i+batch_size]
        batch_results = []
        
        for input_text in batch_inputs:
            # Check cache first
            if cache:
                cached_result = cache.get_cached_response(model_name, input_text)
                if cached_result is not None:
                    batch_results.append(cached_result)
                    cache_hits += 1
                    continue
            
            # Make prediction
            payload = {
                "inputs": input_text,
                "parameters": prediction_params
            }
            
            try:
                response = predictor.predict(payload, custom_attributes="accept_eula=true")
                batch_results.append(response)
                
                # Cache the result
                if cache:
                    cache.cache_response(model_name, input_text, response)
                    
            except Exception as e:
                logger.error(f"Prediction failed for input '{input_text[:50]}...': {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
    
    logger.info(f"Batch prediction completed. Cache hits: {cache_hits}/{len(inputs_list)}")
    return results

def optimize_training_job(estimator, training_data: str, monitor: Optional[PerformanceMonitor] = None):
    """Enhanced training with monitoring and optimization"""
    if monitor is None:
        monitor = PerformanceMonitor()
    
    monitor.start_timing("model_training")
    
    try:
        # Start training
        estimator.fit({"training": training_data})
        
        # Log completion
        training_time = monitor.end_timing("model_training")
        if training_time is not None:
            monitor.log_custom_metric("TrainingTime", training_time, "Seconds")
        
        logger.info("Training completed successfully")
        return estimator
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        monitor.log_custom_metric("TrainingFailures", 1, "Count")
        raise

def print_enhanced_response(payload: Dict[str, Any], response: Any, execution_time: Optional[float] = None):
    """Enhanced response printing with timing information"""
    print(f"Input: {payload['inputs']}")
    print(f"Response: {response}")
    if execution_time is not None:
        print(f"Execution time: {execution_time:.3f} seconds")
    print("\n" + "="*50 + "\n")

def validate_model_performance(predictor, test_prompts: List[str], 
                             expected_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate model performance with automated checks"""
    results = {
        "total_prompts": len(test_prompts),
        "successful_predictions": 0,
        "avg_response_length": 0,
        "keyword_matches": 0,
        "responses": []
    }
    
    total_response_length = 0
    
    for prompt in test_prompts:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 64,
                "top_p": 0.9,
                "temperature": 0.6,
                "return_full_text": False,
            }
        }
        
        try:
            start_time = time.time()
            response = predictor.predict(payload, custom_attributes="accept_eula=true")
            response_time = time.time() - start_time
            
            if response and len(response) > 0:
                results["successful_predictions"] += 1
                response_text = str(response[0].get('generated_text', ''))
                total_response_length += len(response_text)
                
                # Check for expected keywords
                if expected_keywords:
                    for keyword in expected_keywords:
                        if keyword.lower() in response_text.lower():
                            results["keyword_matches"] += 1
                            break
                
                results["responses"].append({
                    "prompt": prompt,
                    "response": response_text,
                    "response_time": response_time
                })
        
        except Exception as e:
            logger.error(f"Validation failed for prompt '{prompt[:50]}...': {e}")
            results["responses"].append({
                "prompt": prompt,
                "response": None,
                "error": str(e)
            })
    
    # Calculate averages
    if results["successful_predictions"] > 0:
        results["avg_response_length"] = total_response_length / results["successful_predictions"]
    
    results["success_rate"] = results["successful_predictions"] / results["total_prompts"]
    
    logger.info(f"Model validation completed. Success rate: {results['success_rate']:.2%}")
    return results

# Global instances for convenience
performance_monitor = PerformanceMonitor()
resource_manager = ResourceManager()
model_cache = ModelCache()