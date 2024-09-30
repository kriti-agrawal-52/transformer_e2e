"""
Model Benchmarking Module for Transformer Models
================================================

This module provides comprehensive benchmarking capabilities for evaluating
model quality, performance, and resource usage across different model variants
(baseline, distilled, quantized, and combined).

WHAT IS MODEL BENCHMARKING?
===========================
Benchmarking involves systematically measuring and comparing different aspects
of model performance to make informed decisions about model deployment.

METRICS IMPLEMENTED:
===================
1. Quality Metrics:
   - Perplexity: How "surprised" the model is by the test data
   - BLEU Score: Quality of generated text compared to references
   - Cross-entropy Loss: Basic prediction accuracy
   - Token Accuracy: Percentage of correctly predicted tokens

2. Performance Metrics:
   - Inference Speed: Tokens generated per second
   - Model Size: Storage requirements in MB
   - Memory Usage: RAM/VRAM consumption during inference
   - CPU/GPU Utilization: Hardware resource usage

3. Resource Metrics:
   - Parameter Count: Number of trainable parameters
   - FLOPs: Floating point operations per forward pass
   - Latency: Time to generate a single token
   - Throughput: Total tokens processed per unit time

DESIGN PHILOSOPHY:
==================
This module uses direct model generation calls for efficient and reliable
text generation evaluation, avoiding subprocess overhead while maintaining
clean code organization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os
import gc
import logging

from typing import Dict, List, Any, Optional, Tuple

import json
from collections import defaultdict
import math

# NLP evaluation libraries
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download('punkt', quiet=True)
except ImportError:
    nltk = None
    
try:
    import sacrebleu
except ImportError:
    sacrebleu = None

# Memory profiling
try:
    from memory_profiler import profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class BenchmarkResults:
    """
    Container for benchmark results from a model evaluation.
    
    This class organizes all metrics into logical groups for easy analysis.
    """
    
    def __init__(self, model_name: str, model_type: str):
        # Model identification
        self.model_name = model_name
        self.model_type = model_type  # "baseline", "distilled", "quantized", "combined"
        
        # Quality metrics
        self.perplexity = 0.0
        self.bleu_score = 0.0
        self.cross_entropy_loss = 0.0
        self.token_accuracy = 0.0
        
        # Performance metrics
        self.inference_speed = 0.0  # tokens per second
        self.latency_per_token = 0.0  # milliseconds per token
        self.total_inference_time = 0.0  # seconds for all samples
        
        # Resource metrics
        self.model_size_mb = 0.0
        self.parameter_count = 0
        self.peak_memory_mb = 0.0
        self.avg_cpu_usage = 0.0
        self.avg_gpu_usage = 0.0
        
        # Generation quality samples
        self.generated_samples = []
        
        # Raw measurement data for statistical analysis
        self.raw_measurements = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "quality_metrics": {
                "perplexity": self.perplexity,
                "bleu_score": self.bleu_score,
                "cross_entropy_loss": self.cross_entropy_loss,
                "token_accuracy": self.token_accuracy
            },
            "performance_metrics": {
                "inference_speed": self.inference_speed,
                "latency_per_token": self.latency_per_token,
                "total_inference_time": self.total_inference_time
            },
            "resource_metrics": {
                "model_size_mb": self.model_size_mb,
                "parameter_count": self.parameter_count,
                "peak_memory_mb": self.peak_memory_mb,
                "avg_cpu_usage": self.avg_cpu_usage,
                "avg_gpu_usage": self.avg_gpu_usage
            },
            "generated_samples": self.generated_samples[:5],  # Save only first 5 samples
            "raw_measurements": self.raw_measurements
        }


class ModelBenchmarker:
    """
    Comprehensive benchmarking system for transformer models.
    
    This class provides a complete suite of evaluation tools to assess
    model quality, performance, and resource usage. It supports comparison
    between different model variants and optimization techniques.
    
    The benchmarker handles:
    - Quality evaluation (perplexity, BLEU, accuracy)
    - Performance measurement (speed, latency, throughput)
    - Resource monitoring (memory, CPU, GPU usage)
    - Statistical analysis and reporting
    
    DESIGN APPROACH:
    ================
    This benchmarker uses direct model generation calls for efficient evaluation,
    avoiding subprocess overhead while providing comprehensive benchmarking
    capabilities for model comparison and optimization.
    """
    
    def __init__(self, config, device: str = "auto"):
        """
        Initialize the model benchmarker.
        
        Args:
            config: Benchmarking configuration SimpleNamespace
            device: Computing device ("auto", "cuda", "cpu")
        """
        self.config = config
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"ModelBenchmarker initialized on device: {self.device}")
        
        # Initialize monitoring
        self.process = psutil.Process()
        
        # Check available evaluation libraries
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check availability of optional dependencies."""
        if nltk is None:
            logger.warning("NLTK not available. BLEU score calculation may be limited.")
        if sacrebleu is None:
            logger.warning("SacreBLEU not available. Using NLTK for BLEU scores.")
        if not MEMORY_PROFILER_AVAILABLE:
            logger.warning("Memory profiler not available. Memory usage will be estimated.")
            
    def benchmark_model(self, model: nn.Module, dataloader, tokenizer,
                       model_name: str, model_type: str) -> BenchmarkResults:
        """
        Perform comprehensive benchmarking of a model.
        
        This is the main entry point for model evaluation. It runs all
        benchmark categories and compiles the results.
        
        Args:
            model: Model to benchmark
            dataloader: Data loader for evaluation
            tokenizer: Tokenizer for text processing
            model_name: Human-readable name for the model
            model_type: Type of model ("baseline", "distilled", etc.)
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting comprehensive benchmarking for {model_name} ({model_type})")
        
        # Initialize results container
        results = BenchmarkResults(model_name=model_name, model_type=model_type)
        
        # Move model to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        try:
            # Run all benchmarking steps
            logger.info(f"Benchmarking {model_name} | Quality, performance, resources, generation...")
            self._evaluate_quality_metrics(model, dataloader, tokenizer, results)
            self._measure_performance_metrics(model, dataloader, results)
            self._analyze_resource_metrics(model, results)
            self._evaluate_generation_quality_direct(model, tokenizer, results)
            
            logger.info(f"Benchmarking completed | {model_name} | Perplexity: {results.perplexity:.2f} | Speed: {results.inference_speed:.1f} tok/s")
            return results
            
        except Exception as e:
            logger.error(f"Benchmarking failed for {model_name}: {str(e)}")
            raise
            
    def _evaluate_quality_metrics(self, model: nn.Module, dataloader,
                                 tokenizer, results: BenchmarkResults):
        """
        Evaluate model quality using various NLP metrics.
        
        Args:
            model: Model to evaluate
            dataloader: Data for evaluation
            tokenizer: Tokenizer for processing
            results: Results container to update
        """
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        log_probabilities = []
        
        # Use a subset of data for efficiency
        samples_processed = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                if samples_processed >= self.config.perplexity_samples:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get model predictions
                outputs = model(inputs)
                
                # Calculate cross-entropy loss
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1),
                    reduction='sum'
                )
                
                # Accumulate metrics
                total_loss += loss.item()
                batch_tokens = targets.numel()
                total_tokens += batch_tokens
                
                # Calculate token accuracy
                predictions = outputs.argmax(dim=-1)
                correct_predictions += (predictions == targets).sum().item()
                
                # Collect log probabilities for perplexity
                log_probs = F.log_softmax(outputs, dim=-1)
                target_log_probs = log_probs.gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                log_probabilities.extend(target_log_probs.cpu().numpy().flatten())
                
                samples_processed += inputs.size(0)
                
        # Calculate final metrics
        if total_tokens > 0:
            results.cross_entropy_loss = total_loss / total_tokens
            results.token_accuracy = correct_predictions / total_tokens
            
            # Calculate perplexity
            # Perplexity = exp(average negative log likelihood)
            avg_log_prob = np.mean(log_probabilities)
            results.perplexity = math.exp(-avg_log_prob)
            
        logger.info(f"Quality metrics - Perplexity: {results.perplexity:.2f}, "
                   f"Token Accuracy: {results.token_accuracy:.3f}")
                   
    def _measure_performance_metrics(self, model: nn.Module, dataloader,
                                   results: BenchmarkResults):
        """
        Measure model performance including speed and latency.
        
        Args:
            model: Model to measure
            dataloader: Data for measurement
            results: Results container to update
        """
        # Warmup runs to stabilize measurements
        logger.debug("Running warmup iterations...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= self.config.warmup_steps:
                    break
                inputs = inputs.to(self.device)
                _ = model(inputs)
                
        # Clear cache and collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Actual performance measurement
        inference_times = []
        tokens_processed = 0
        
        logger.debug("Measuring inference performance...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= self.config.speed_samples:
                    break
                    
                inputs = inputs.to(self.device)
                batch_tokens = inputs.numel()
                
                # Measure inference time
                start_time = time.perf_counter()
                _ = model(inputs)
                
                # Synchronize GPU if available
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                end_time = time.perf_counter()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                tokens_processed += batch_tokens
                
        # Calculate performance metrics
        if inference_times:
            total_time = sum(inference_times)
            avg_time_per_batch = np.mean(inference_times)
            
            results.total_inference_time = total_time
            results.inference_speed = tokens_processed / total_time  # tokens per second
            results.latency_per_token = (total_time / tokens_processed) * 1000  # ms per token
            
            # Store raw measurements for statistical analysis
            results.raw_measurements["inference_times"] = inference_times
            results.raw_measurements["tokens_per_batch"] = [
                inputs.numel() for inputs, _ in dataloader
            ][:len(inference_times)]
            
        logger.info(f"Performance metrics - Speed: {results.inference_speed:.1f} tokens/sec, "
                   f"Latency: {results.latency_per_token:.2f} ms/token")
                   
    def _analyze_resource_metrics(self, model: nn.Module, results: BenchmarkResults):
        """
        Analyze model resource usage including size and memory.
        
        Args:
            model: Model to analyze
            results: Results container to update
        """
        # Model size calculation
        results.parameter_count = sum(p.numel() for p in model.parameters())
        
        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        results.model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        # Memory usage measurement
        if self.config.monitor_memory:
            memory_measurements = []
            
            # Measure memory during inference
            for i in range(self.config.memory_samples):
                # Get current memory usage
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    memory_measurements.append(gpu_memory)
                else:
                    ram_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
                    memory_measurements.append(ram_memory)
                    
                # Small delay to get different measurements
                time.sleep(0.1)
                
            if memory_measurements:
                results.peak_memory_mb = max(memory_measurements)
                results.raw_measurements["memory_usage"] = memory_measurements
                
        # CPU usage (approximate)
        if self.config.monitor_cpu:
            cpu_measurements = []
            for _ in range(5):  # Quick sampling
                cpu_usage = psutil.cpu_percent(interval=0.1)
                cpu_measurements.append(cpu_usage)
            results.avg_cpu_usage = np.mean(cpu_measurements)
            results.raw_measurements["cpu_usage"] = cpu_measurements
            
        # GPU usage (if available)
        if self.config.monitor_gpu and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                results.avg_gpu_usage = gpu_util.gpu
            except ImportError:
                logger.warning("pynvml not available for GPU monitoring")
            except Exception as e:
                logger.warning(f"GPU monitoring failed: {str(e)}")
                
        logger.info(f"Resource metrics - Model size: {results.model_size_mb:.1f} MB, "
                   f"Parameters: {results.parameter_count:,}, "
                   f"Peak memory: {results.peak_memory_mb:.1f} MB")
                   

    def _evaluate_generation_quality_direct(self, model: nn.Module, tokenizer,
                                           results: BenchmarkResults):
        """
        Evaluate text generation quality using direct model generation calls.
        
        This method uses the model's generate method directly rather than 
        subprocess calls, making it simpler and more reliable.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for encoding/decoding
            results: Results container to update
        """
        if not (nltk or sacrebleu):
            logger.warning("No BLEU evaluation library available. Skipping BLEU score calculation.")
            return
            
        generated_texts = []
        bleu_scores = []
        generation_times = []
        
        model.eval()
        with torch.no_grad():
            # Generate text samples for evaluation
            for i, prompt in enumerate(self.config.eval_prompts):
                if i >= self.config.bleu_samples:
                    break
                    
                try:
                    # Encode prompt
                    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                    
                    # Generate text using model's generate method
                    start_time = time.perf_counter()
                    
                    generated_ids = model.generate(
                        input_ids,
                        max_new_tokens=self.config.generation_length,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        device=self.device
                    )
                    
                    end_time = time.perf_counter()
                    generation_time = end_time - start_time
                    generation_times.append(generation_time)
                    
                    # Decode generated text
                    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)
                    
                    # Remove the original prompt from generated text to get only new content
                    if generated_text.startswith(prompt):
                        new_content = generated_text[len(prompt):].strip()
                    else:
                        new_content = generated_text.strip()
                    
                    full_output = f"Prompt: {prompt}\nGenerated: {new_content}"
                    generated_texts.append(full_output)
                    
                    # Calculate BLEU score if libraries available
                    if nltk:
                        # Use prompt as reference (simplified evaluation)
                        reference = prompt.split()
                        candidate = new_content.split()
                        
                        # Calculate BLEU score with smoothing
                        smoothing = SmoothingFunction().method4
                        bleu_score = sentence_bleu(
                            [reference], candidate, 
                            smoothing_function=smoothing
                        )
                        bleu_scores.append(bleu_score)
                        
                except Exception as e:
                    logger.warning(f"Text generation failed for prompt '{prompt}': {str(e)}")
                    continue
                    
        # Calculate average BLEU score
        if bleu_scores:
            results.bleu_score = np.mean(bleu_scores)
            results.raw_measurements["bleu_scores"] = bleu_scores
            
        # Store sample generations
        results.generated_samples = generated_texts[:5]  # Keep only first 5 samples
        
        # Store generation performance data
        if generation_times:
            results.raw_measurements["generation_times"] = generation_times
            avg_generation_time = np.mean(generation_times)
            logger.info(f"Average generation time: {avg_generation_time:.2f}s per sample")
            
        logger.info(f"Generation quality - BLEU score: {results.bleu_score:.3f}, "
                   f"Generated {len(generated_texts)} samples")
                   
    def compare_models(self, results_list: List[BenchmarkResults]) -> Dict[str, Any]:
        """
        Compare multiple model benchmark results.
        
        This method creates a comprehensive comparison between different
        model variants, highlighting trade-offs and relative performance.
        
        Args:
            results_list: List of benchmark results to compare
            
        Returns:
            Dictionary containing comparison analysis
        """
        if not results_list:
            return {}
            
        logger.info(f"Comparing {len(results_list)} model variants")
        
        # Find baseline model for relative comparisons
        baseline = None
        for result in results_list:
            if result.model_type == "baseline":
                baseline = result
                break
                
        comparison = {
            "models_compared": len(results_list),
            "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics_comparison": {},
            "rankings": {},
            "trade_offs": {},
            "summary": {}
        }
        
        # Extract metrics for comparison
        metrics = {
            "perplexity": [r.perplexity for r in results_list],
            "bleu_score": [r.bleu_score for r in results_list],
            "inference_speed": [r.inference_speed for r in results_list],
            "model_size_mb": [r.model_size_mb for r in results_list],
            "peak_memory_mb": [r.peak_memory_mb for r in results_list],
            "token_accuracy": [r.token_accuracy for r in results_list]
        }
        
        model_names = [r.model_name for r in results_list]
        
        # Create detailed comparisons
        for metric_name, values in metrics.items():
            comparison["metrics_comparison"][metric_name] = {
                "values": dict(zip(model_names, values)),
                "best_model": model_names[np.argmin(values) if metric_name == "perplexity" 
                                       else np.argmax(values)],
                "worst_model": model_names[np.argmax(values) if metric_name == "perplexity" 
                                        else np.argmin(values)],
                "range": max(values) - min(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
            
            # Relative to baseline if available
            if baseline:
                baseline_value = getattr(baseline, metric_name)
                if baseline_value > 0:
                    relative_values = [(v / baseline_value - 1) * 100 for v in values]
                    comparison["metrics_comparison"][metric_name]["relative_to_baseline"] = dict(
                        zip(model_names, relative_values)
                    )
                    
        # Create rankings
        for metric_name, values in metrics.items():
            # For perplexity, lower is better; for others, higher is better
            ascending = metric_name in ["perplexity", "model_size_mb", "peak_memory_mb"]
            ranked_indices = np.argsort(values)
            if not ascending:
                ranked_indices = ranked_indices[::-1]
                
            comparison["rankings"][metric_name] = [
                model_names[i] for i in ranked_indices
            ]
            
        # Analyze trade-offs
        comparison["trade_offs"] = self._analyze_trade_offs(results_list)
        
        # Generate summary insights
        comparison["summary"] = self._generate_comparison_summary(results_list, comparison)
        
        return comparison
        
    def _analyze_trade_offs(self, results_list: List[BenchmarkResults]) -> Dict[str, Any]:
        """
        Analyze trade-offs between different metrics.
        
        Args:
            results_list: List of benchmark results
            
        Returns:
            Dictionary containing trade-off analysis
        """
        trade_offs = {
            "size_vs_quality": [],
            "speed_vs_quality": [],
            "memory_vs_quality": [],
            "efficiency_scores": {}
        }
        
        for result in results_list:
            # Size vs Quality trade-off
            size_quality_ratio = result.model_size_mb / (result.token_accuracy + 0.001)
            trade_offs["size_vs_quality"].append({
                "model": result.model_name,
                "size_mb": result.model_size_mb,
                "accuracy": result.token_accuracy,
                "ratio": size_quality_ratio
            })
            
            # Speed vs Quality trade-off
            speed_quality_ratio = result.inference_speed * result.token_accuracy
            trade_offs["speed_vs_quality"].append({
                "model": result.model_name,
                "speed": result.inference_speed,
                "accuracy": result.token_accuracy,
                "ratio": speed_quality_ratio
            })
            
            # Memory vs Quality trade-off
            memory_quality_ratio = result.peak_memory_mb / (result.token_accuracy + 0.001)
            trade_offs["memory_vs_quality"].append({
                "model": result.model_name,
                "memory_mb": result.peak_memory_mb,
                "accuracy": result.token_accuracy,
                "ratio": memory_quality_ratio
            })
            
            # Overall efficiency score (higher is better)
            # Combines accuracy, speed, and inverse of size
            efficiency_score = (
                result.token_accuracy * 
                np.log(result.inference_speed + 1) * 
                np.log(100 / (result.model_size_mb + 1))
            )
            trade_offs["efficiency_scores"][result.model_name] = efficiency_score
            
        return trade_offs
        
    def _generate_comparison_summary(self, results_list: List[BenchmarkResults], 
                                   comparison: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate human-readable summary of model comparison.
        
        Args:
            results_list: List of benchmark results
            comparison: Comparison analysis
            
        Returns:
            Dictionary containing summary insights
        """
        summary = {}
        
        # Find best models for different use cases
        rankings = comparison["rankings"]
        
        summary["best_quality"] = rankings["token_accuracy"][0]
        summary["fastest"] = rankings["inference_speed"][0]
        summary["smallest"] = rankings["model_size_mb"][0]
        summary["most_efficient"] = max(
            comparison["trade_offs"]["efficiency_scores"],
            key=comparison["trade_offs"]["efficiency_scores"].get
        )
        
        # Generate recommendations
        recommendations = []
        
        # Quality recommendation
        best_quality_model = summary["best_quality"]
        recommendations.append(
            f"For best quality: {best_quality_model} with "
            f"{comparison['metrics_comparison']['token_accuracy']['values'][best_quality_model]:.3f} accuracy"
        )
        
        # Speed recommendation
        fastest_model = summary["fastest"]
        recommendations.append(
            f"For fastest inference: {fastest_model} with "
            f"{comparison['metrics_comparison']['inference_speed']['values'][fastest_model]:.1f} tokens/sec"
        )
        
        # Size recommendation
        smallest_model = summary["smallest"]
        recommendations.append(
            f"For smallest size: {smallest_model} with "
            f"{comparison['metrics_comparison']['model_size_mb']['values'][smallest_model]:.1f} MB"
        )
        
        # Balanced recommendation
        most_efficient = summary["most_efficient"]
        recommendations.append(
            f"For best balance: {most_efficient} with efficiency score "
            f"{comparison['trade_offs']['efficiency_scores'][most_efficient]:.2f}"
        )
        
        summary["recommendations"] = recommendations
        
        return summary
        
    def save_benchmark_results(self, results: BenchmarkResults, save_path: str):
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results to save
            save_path: Path to save the results
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
            
        logger.info(f"Benchmark results saved to {save_path}")
        
    def save_comparison_results(self, comparison: Dict[str, Any], save_path: str):
        """
        Save model comparison results to file.
        
        Args:
            comparison: Comparison results to save
            save_path: Path to save the comparison
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
            
        logger.info(f"Comparison results saved to {save_path}")
        
    def benchmark_all_variants(self, models_info: Dict[str, Dict[str, Any]], 
                              dataloader, tokenizer, save_dir: str) -> Dict[str, BenchmarkResults]:
        """
        Benchmark all model variants and create comparison.
        
        This is a convenience method that benchmarks multiple models,
        compares them, and saves all results.
        
        Args:
            models_info: Dictionary mapping model names to model info dicts
                        Each info dict should contain: 'model', 'checkpoint_path', 'run_id'
            dataloader: Data loader for evaluation
            tokenizer: Tokenizer for text processing
            save_dir: Directory to save results
            
        Returns:
            Dictionary mapping model names to their benchmark results
        """
        logger.info(f"Benchmarking {len(models_info)} model variants")
        
        all_results = {}
        results_list = []
        
        for model_name, model_info in models_info.items():
            logger.info(f"Benchmarking {model_name}...")
            
            try:
                # Extract model information
                model = model_info['model']
                checkpoint_path = model_info['checkpoint_path']
                run_id = model_info['run_id']
                
                # Determine model type from name
                model_type = "baseline"
                if "distilled" in model_name.lower():
                    model_type = "distilled"
                elif "quantized" in model_name.lower():
                    model_type = "quantized"
                elif "combined" in model_name.lower():
                    model_type = "combined"
                    
                # Benchmark the model
                results = self.benchmark_model(
                    model, dataloader, tokenizer, model_name, model_type,
                    checkpoint_path, run_id
                )
                
                all_results[model_name] = results
                results_list.append(results)
                
                # Save individual results
                self.save_benchmark_results(
                    results, 
                    os.path.join(save_dir, f"{model_name}_benchmark.json")
                )
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {str(e)}")
                continue
                
        # Create and save comparison
        if len(results_list) > 1:
            logger.info("Creating model comparison...")
            
            comparison = self.compare_models(results_list)
            self.save_comparison_results(
                comparison,
                os.path.join(save_dir, "model_comparison.json")
            )
            
            # Log summary
            if "summary" in comparison and "recommendations" in comparison["summary"]:
                logger.info("Benchmarking recommendations:")
                for recommendation in comparison["summary"]["recommendations"]:
                    logger.info(f"  • {recommendation}")
                    
        logger.info(f"Benchmarking completed | Results saved to {save_dir}")
        return all_results 