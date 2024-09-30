"""
Model Variants Manager for Transformer Models
=============================================

This module orchestrates the creation and management of the four model variants:
1. Baseline - Original trained model
2. Distilled - Knowledge distilled smaller model  
3. Quantized - INT8 quantized model
4. Combined - Both distilled and quantized model

WHAT IS A MODEL VARIANT?
========================
A model variant is a version of the original model that has been optimized
for different deployment scenarios. Each variant offers different trade-offs
between accuracy, speed, and resource usage.

THE FOUR VARIANTS:
=================
1. BASELINE: The original, unmodified trained model
   - Highest accuracy
   - Largest size and memory usage
   - Slowest inference
   - Best for scenarios where accuracy is paramount

2. DISTILLED: Knowledge distilled into a smaller architecture
   - Good accuracy (85-90% of baseline)
   - Medium size and memory usage
   - Medium inference speed
   - Good balance for most deployment scenarios

3. QUANTIZED: Original model with reduced precision (INT8)
   - Good accuracy (90-95% of baseline)
   - Smallest size (4x compression)
   - Fastest inference
   - Best for edge devices and mobile deployment

4. COMBINED: Both distilled and quantized
   - Moderate accuracy (80-85% of baseline)
   - Very small size and memory usage
   - Very fast inference
   - Best for extremely resource-constrained environments

WORKFLOW:
========
1. Load baseline model from W&B
2. Create distilled variants using knowledge distillation
3. Create quantized variants using INT8 quantization
4. Create combined variants (distill then quantize)
5. Benchmark all variants for comparison
6. Generate deployment recommendations
"""

import torch
import torch.nn as nn
import logging
import wandb
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import copy
import traceback
import threading
import glob
import time

from src.packaging.distillation import ModelDistiller
from src.packaging.quantization import ModelQuantizer
from src.packaging.quantization_aware_distillation import QuantizationAwareDistiller
from src.packaging.benchmarking import ModelBenchmarker, BenchmarkResults
from src.models.transformer import TransformerModel
from src.utils.config_loader import load_config
from src.data.processing import setup_data_and_tokenizer, PreprocessingTraining

logger = logging.getLogger(__name__)


class ModelVariant:
    """
    Container for a model variant with its metadata.
    
    This class encapsulates all information about a specific model variant,
    including the model itself, performance metrics, and deployment metadata.
    """
    
    def __init__(self, name: str, variant_type: str, model: nn.Module,
                 checkpoint_path: str, run_id: str):
        """
        Initialize a model variant.
        
        Args:
            name: Human-readable name for the variant
            variant_type: Type ("baseline", "distilled", "quantized", "combined")
            model: The actual model instance
            checkpoint_path: Path to the saved checkpoint
            run_id: W&B run ID associated with this variant
        """
        self.name = name
        self.variant_type = variant_type
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.run_id = run_id
        
        # Performance metrics (filled during benchmarking)
        self.benchmark_results: Optional[BenchmarkResults] = None
        
        # Model characteristics
        self.parameter_count = sum(p.numel() for p in model.parameters())
        self.model_size_mb = self._calculate_model_size()
        
        # Optimization metadata
        self.optimization_info = {
            "variant_type": variant_type,
            "parameter_count": self.parameter_count,
            "model_size_mb": self.model_size_mb,
            "checkpoint_path": checkpoint_path,
            "run_id": run_id
        }
        
    def _calculate_model_size(self) -> float:
        """Calculate model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
        
    def get_compression_ratio(self, baseline_size_mb: float) -> float:
        """
        Calculate compression ratio compared to baseline.
        
        Args:
            baseline_size_mb: Size of the baseline model in MB
            
        Returns:
            Compression ratio (baseline_size / this_size)
        """
        if self.model_size_mb > 0:
            return baseline_size_mb / self.model_size_mb
        return 1.0
        
    def get_speedup_ratio(self, baseline_speed: float) -> float:
        """
        Calculate inference speedup compared to baseline.
        
        Args:
            baseline_speed: Inference speed of the baseline model
            
        Returns:
            Speedup ratio (this_speed / baseline_speed)
        """
        if self.benchmark_results and baseline_speed > 0:
            return self.benchmark_results.inference_speed / baseline_speed
        return 1.0
        
    def get_accuracy_retention(self, baseline_accuracy: float) -> float:
        """
        Calculate accuracy retention compared to baseline.
        
        Args:
            baseline_accuracy: Accuracy of the baseline model
            
        Returns:
            Accuracy retention ratio (this_accuracy / baseline_accuracy)
        """
        if self.benchmark_results and baseline_accuracy > 0:
            return self.benchmark_results.token_accuracy / baseline_accuracy
        return 1.0
        
    def to_dict(self, baseline_info: Optional[Dict[str, Any]] = None, baseline_benchmark: Optional[BenchmarkResults] = None) -> Dict[str, Any]:
        """Convert variant to dictionary for serialization."""
        result = {
            "name": self.name,
            "variant_type": self.variant_type,
            "parameter_count": self.parameter_count,
            "model_size_mb": self.model_size_mb,
            "checkpoint_path": self.checkpoint_path,
            "run_id": self.run_id,
            "optimization_info": self.optimization_info
        }
        
        # Add performance ratios if baseline info is provided
        if baseline_info:
            result["performance_ratios"] = {
                "compression_ratio": self.get_compression_ratio(baseline_info["model_size_mb"]),
                "speedup_ratio": self.get_speedup_ratio(baseline_benchmark.inference_speed) if baseline_benchmark else None,
                "accuracy_retention": self.get_accuracy_retention(baseline_benchmark.token_accuracy) if baseline_benchmark else None
            }
        
        if self.benchmark_results:
            result["benchmark_results"] = self.benchmark_results.to_dict()
            
        return result


class ModelVariantsManager:
    """
    Master orchestrator for creating and managing all model variants.
    
    This class coordinates the entire packaging pipeline:
    1. Loading the baseline model from W&B
    2. Creating optimized variants through distillation and quantization
    3. Benchmarking all variants for performance comparison
    4. Generating deployment recommendations
    5. Saving all artifacts and results
    
    The manager ensures a consistent workflow and provides comprehensive
    reporting on the trade-offs between different optimization techniques.
    """
    
    def __init__(self, config):
        """
        Initialize the model variants manager.
        
        Args:
            config: Packaging configuration SimpleNamespace
        """
        self.config = config
        
        # Auto-detect device
        if config.compute.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.compute.device)
            
        logger.info(f"ModelVariantsManager initialized on device: {self.device}")
        
        # Initialize optimizers
        self.distiller = ModelDistiller(config.optimization.distillation, str(self.device))
        self.quantizer = ModelQuantizer(config.optimization.quantization, str(self.device))
        
        # Initialize QAD module if combined config is available
        self.combined_config = getattr(config.optimization, 'combined', None)
        if self.combined_config:
            self.qad_distiller = QuantizationAwareDistiller(self.combined_config, str(self.device))
        else:
            self.qad_distiller = None
        
        self.benchmarker = ModelBenchmarker(config.benchmarking, str(self.device))
        
        # Storage for variants
        self.variants: Dict[str, ModelVariant] = {}
        
        # Data preprocessing components (will be initialized when needed)
        self.preprocessor = None
        self.tokenizer = None
        
        # Base model will be initialized during pipeline setup
        self.base_model = None
        
        # Create output directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary output directories."""
        directories = [
            self.config.output.output_dir,
            os.path.join(self.config.output.output_dir, self.config.output.results_dir),
            os.path.join(self.config.output.output_dir, "distilled"),
            os.path.join(self.config.output.output_dir, "quantized"),
            os.path.join(self.config.output.output_dir, "combined")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info(f"Output directories created under: {self.config.output.output_dir}")
    
    def _calculate_student_architectures(self, base_model_config: Dict[str, Any]) -> None:
        """
        Calculate student architectures based on base model configuration.
        
        Args:
            base_model_config: Base model configuration dictionary
        """
        logger.info("Calculating student architectures from base model...")
        
        # Store calculated architectures for use during distillation
        self.calculated_student_architectures = {}
        
        for student_name in dir(self.config.optimization.distillation.student_architectures):
            if not student_name.startswith('_'):
                ratios = getattr(self.config.optimization.distillation.student_architectures, student_name)
                calculated_arch = {
                    "num_layers": max(1, int(base_model_config["num_layers"] * ratios.num_layers_ratio)),
                    "num_heads": max(1, int(base_model_config["num_heads"] * ratios.num_heads_ratio)),
                    "channel_dim": max(64, int(base_model_config["channel_dim"] * ratios.channel_dim_ratio)),
                    "context_window": base_model_config["context_window"],  # Keep same context window
                    "dropout_rate": base_model_config["dropout_rate"]  # Keep same dropout
                }
                
                # Ensure channel_dim is divisible by num_heads for attention
                if calculated_arch["channel_dim"] % calculated_arch["num_heads"] != 0:
                    calculated_arch["channel_dim"] = (
                        (calculated_arch["channel_dim"] // calculated_arch["num_heads"]) * 
                        calculated_arch["num_heads"]
                    )
                
                self.calculated_student_architectures[student_name] = calculated_arch
                
                logger.info(f"Student '{student_name}': {calculated_arch['num_layers']} layers, "
                           f"{calculated_arch['num_heads']} heads, {calculated_arch['channel_dim']} dim")
        
    def load_baseline_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """
        Load the baseline model from W&B.
        
        Args:
            model_config: Model configuration parameters
            
        Returns:
            The loaded baseline model (not a variant)
        """
        logger.info(f"Loading baseline model from W&B run: {self.config.wandb.source_run_id}")
        
        try:
            # Initialize W&B for downloading model
            wandb.init(
                project=self.config.wandb.project,
                job_type="model_loading"
            )
            
            # Download model artifact from W&B
            # Check if artifact name already has a version, if not append :latest
            artifact_name = self.config.wandb.source_run_id
            if ':' not in artifact_name:
                artifact_name = f"{artifact_name}:latest"
            artifact = wandb.use_artifact(artifact_name)
            artifact_dir = artifact.download()
            
            # Find the model checkpoint
            checkpoint_path = None
            for file in os.listdir(artifact_dir):
                if file.endswith('.pt') or file.endswith('.pth'):
                    checkpoint_path = os.path.join(artifact_dir, file)
                    break
                    
            if not checkpoint_path:
                raise FileNotFoundError(f"No model checkpoint found in W&B artifact")
                
            logger.info(f"Found checkpoint: {checkpoint_path}")
            
            # Load the model
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create model instance
            baseline_model = TransformerModel(
                vocab_size=model_config["vocab_size"],
                channel_dim=model_config["channel_dim"],
                context_window=model_config["context_window"],
                num_heads=model_config["num_heads"],
                num_layers=model_config["num_layers"],
                dropout_rate=model_config.get("dropout_rate", 0.2),
                final_dropout_multiplier=model_config.get("final_dropout_multiplier"),
                max_dropout_val=model_config.get("max_dropout_val", 0.5)
            ).to(self.device)
            
            # Load state dict
            baseline_model.load_state_dict(checkpoint["model_state_dict"])
            baseline_model.eval()
            
            # Store baseline model info for reference
            self.baseline_info = {
                "parameter_count": sum(p.numel() for p in baseline_model.parameters()),
                "model_size_mb": sum(p.numel() * p.element_size() for p in baseline_model.parameters()) / (1024 * 1024),
                "checkpoint_path": checkpoint_path,
                "run_id": self.config.wandb.source_run_id
            }
            
            logger.info(f"Baseline model loaded | Params: {self.baseline_info['parameter_count']:,} | Size: {self.baseline_info['model_size_mb']:.1f} MB")
            
            return baseline_model
            
        except Exception as e:
            logger.error(f"Failed to load baseline model: {str(e)}")
            raise
            
    def create_distilled_variants(self, train_dataloader, val_dataloader) -> Dict[str, ModelVariant]:
        """
        Create distilled model variants using knowledge distillation.
        
        Args:
            train_dataloader: Training data for distillation
            val_dataloader: Validation data for evaluation
            
        Returns:
            Dictionary of distilled model variants
        """
        if not self.config.optimization.methods.create_distilled:
            logger.info("Distillation disabled in configuration")
            return {}
            
        logger.info("Creating distilled model variants...")
        
        # Use calculated student architectures
        if not hasattr(self, 'calculated_student_architectures'):
            raise RuntimeError("Student architectures not calculated. Run pipeline setup first.")
        
        # Set the teacher model using the base model
        self.distiller.set_teacher_model(self.base_model)
        
        # Create student models using calculated architectures
        self.distiller.create_student_models_from_architectures(
            self.calculated_student_architectures, 
            self.tokenizer.vocab_size
        )
        
        # Distill all student architectures
        distillation_results = self.distiller.distill_all_students(
            train_dataloader,
            val_dataloader,
            os.path.join(self.config.output.output_dir, "distilled")
        )
        
        # Create variants from distillation results
        distilled_variants = {}
        
        for student_name, results in distillation_results.items():
            if "error" in results:
                logger.warning(f"Skipping failed distillation: {student_name}")
                continue
                
            # Log training metrics to W&B and get run ID
            wandb_run_id = self._log_distillation_metrics_to_wandb(
                student_name, results, "knowledge_distillation"
            )
                
            # Get the distilled model
            distilled_model = self.distiller.student_models[student_name]
            
            # Create checkpoint path
            checkpoint_path = os.path.join(
                self.config.output.output_dir, "distilled", f"{student_name}_best.pt"
            )
            
            # Create variant
            variant = ModelVariant(
                name=f"Distilled ({student_name.capitalize()})",
                variant_type="distilled",
                model=distilled_model,
                checkpoint_path=checkpoint_path,
                run_id=wandb_run_id  # Use the actual W&B run ID
            )
            
            # Add distillation-specific metadata
            variant.optimization_info.update({
                "distillation_results": results,
                "student_architecture": self.calculated_student_architectures[student_name],
                "compression_ratio": variant.get_compression_ratio(self.baseline_info["model_size_mb"]),
                "wandb_run_id": wandb_run_id
            })
            
            distilled_variants[f"distilled_{student_name}"] = variant
            self.variants[f"distilled_{student_name}"] = variant
            
        logger.info(f"Created {len(distilled_variants)} distilled variants")
        return distilled_variants
        
    def create_quantized_variants(self, train_dataloader, val_dataloader) -> Dict[str, ModelVariant]:
        """
        Create quantized model variants using INT8 quantization.
        
        Args:
            train_dataloader: Training data for QAT
            val_dataloader: Validation data for calibration
            
        Returns:
            Dictionary of quantized model variants
        """
        if not self.config.optimization.methods.create_quantized:
            logger.info("Quantization disabled in configuration")
            return {}
            
        logger.info("Creating quantized model variants...")
        
        # Create quantized variants of the base model
        quantization_results = self.quantizer.quantize_model_variants(
            copy.deepcopy(self.base_model),
            train_dataloader,
            val_dataloader,
            os.path.join(self.config.output.output_dir, "quantized")
        )
        
        # Create variants from quantization results
        quantized_variants = {}
        
        for quant_method, results in quantization_results.items():
            if not results.get("success", False):
                logger.warning(f"Skipping failed quantization: {quant_method}")
                continue
                
            # Get the quantized model
            quantized_model = results["model"]
            
            # Create checkpoint path
            checkpoint_path = os.path.join(
                self.config.output.output_dir, "quantized", f"{quant_method}_quantized.pt"
            )
            
            # Create run ID for this variant
            run_id = f"{self.config.wandb.source_run_id}_quantized_{quant_method}"
            
            # Create variant
            variant = ModelVariant(
                name=f"Quantized ({quant_method.upper()})",
                variant_type="quantized",
                model=quantized_model,
                checkpoint_path=checkpoint_path,
                run_id=run_id
            )
            
            # Add quantization-specific metadata
            variant.optimization_info.update({
                "quantization_method": quant_method,
                "quantization_results": {k: v for k, v in results.items() if k != "model"},
                "compression_ratio": variant.get_compression_ratio(self.baseline_info["model_size_mb"])
            })
            
            quantized_variants[f"quantized_{quant_method}"] = variant
            self.variants[f"quantized_{quant_method}"] = variant
            
        logger.info(f"Created {len(quantized_variants)} quantized variants")
        return quantized_variants
        
    def create_combined_variants(self, train_dataloader, val_dataloader) -> Dict[str, ModelVariant]:
        """
        Create combined variants using quantization-aware distillation.
        
        This method performs true quantization-aware distillation where student models
        learn with quantization simulation during the distillation process, resulting
        in better accuracy than sequential distillation → quantization.
        
        Args:
            train_dataloader: Training data for QAD
            val_dataloader: Validation data for evaluation
            
        Returns:
            Dictionary of combined model variants
        """
        if not self.config.optimization.methods.create_combined:
            logger.info("Combined variants disabled in configuration")
            return {}
            
        logger.info("Creating combined variants using quantization-aware distillation...")
        
        if not self.qad_distiller:
            raise ValueError("QAD module not initialized. Combined config missing.")
        
        # Set the teacher model for QAD
        self.qad_distiller.set_teacher_model(self.base_model)
        
        # Create student models for QAD using calculated architectures
        self.qad_distiller.create_student_models_from_architectures(
            self.calculated_student_architectures, 
            self.tokenizer.vocab_size
        )
        
        # Prepare quantization configuration for QAD
        quantization_config = {
            'backend': getattr(self.combined_config.quantization, 'backend', 'fbgemm')
        }
            
        combined_variants = {}
        
        # Perform quantization-aware distillation for each student architecture
        for student_name in self.calculated_student_architectures.keys():
            logger.info(f"Creating QAD variant: {student_name}")
            
            try:
                # Perform quantization-aware distillation using dedicated QAD module
                qad_results = self.qad_distiller.distill_student_with_quantization(
                    student_name,
                    train_dataloader,
                    val_dataloader,
                    os.path.join(self.config.output.output_dir, "combined"),
                    quantization_config
                )
                
                if "error" in qad_results:
                    logger.warning(f"Skipping failed QAD: {student_name}")
                    continue

                # Log QAD training metrics to W&B and get run ID
                wandb_run_id = self._log_distillation_metrics_to_wandb(
                    student_name, qad_results, "quantization_aware_distillation"
                )
                
                # Get the quantized model from results
                quantized_model = qad_results["model"]
                
                # Create checkpoint path
                checkpoint_path = os.path.join(
                    self.config.output.output_dir, "combined", f"{student_name}_qad_best.pt"
                )

                # Create variant
                variant_name = f"combined_qad_{student_name}"
                variant = ModelVariant(
                    name=f"QAD ({student_name.capitalize()})",
                    variant_type="combined",
                    model=quantized_model,
                    checkpoint_path=checkpoint_path,
                    run_id=wandb_run_id  # Use the actual W&B run ID
                )

                # Add QAD-specific metadata
                variant.optimization_info.update({
                    "optimization_method": "quantization_aware_distillation",
                    "student_architecture": self.calculated_student_architectures[student_name],
                    "qad_results": {k: v for k, v in qad_results.items() if k != "model"},
                    "compression_ratio": variant.get_compression_ratio(self.baseline_info["model_size_mb"]),
                    "quantization_backend": quantization_config['backend'],
                    "wandb_run_id": wandb_run_id
                })

                combined_variants[variant_name] = variant
                self.variants[variant_name] = variant

                logger.info(f"✓ Created QAD variant: {variant.name} | Compression: {variant.optimization_info['compression_ratio']:.2f}x")
                
            except Exception as e:
                logger.error(f"Failed to create QAD variant for {student_name}: {str(e)}")
                continue
                
        logger.info(f"Created {len(combined_variants)} quantization-aware distilled variants")
        return combined_variants
        
    def benchmark_all_variants(self, test_loader) -> Dict[str, BenchmarkResults]:
        """
        Benchmark all created variants.
        
        Args:
            test_loader: Test data loader function
            
        Returns:
            Dictionary mapping variant names to benchmark results
        """
        if not self.variants:
            raise ValueError("No variants available for benchmarking")
        
        logger.info(f"Benchmarking {len(self.variants)} model variants...")
        
        # Prepare models info for benchmarker
        models_info = {}
        for name, variant in self.variants.items():
            models_info[name] = {
                'model': variant.model,
                'name': variant.name,
                'type': variant.variant_type
            }
        
        # Use internal benchmarker
        results_dir = os.path.join(self.config.output.output_dir, self.config.output.results_dir)
        benchmark_results = self.benchmarker.benchmark_all_variants(
            models_info,
            test_loader,
            self.tokenizer,
            results_dir
        )
        
        logger.info("Benchmarking completed successfully")
        return benchmark_results
        
    def generate_deployment_recommendations(self) -> Dict[str, Any]:
        """
        Generate deployment recommendations based on benchmark results.
        
        Returns:
            Dictionary containing deployment recommendations for different scenarios
        """
        logger.info("Generating deployment recommendations...")
        
        if not self.variants:
            return {"error": "No variants available for recommendations"}
            
        # Get baseline for comparisons
        baseline = self.variants.get("baseline")
        if not baseline or not baseline.benchmark_results:
            return {"error": "Baseline variant not found or not benchmarked"}
            
        recommendations = {
            "scenarios": {},
            "variant_analysis": {},
            "summary": {}
        }
        
        # Analyze each variant
        for variant_name, variant in self.variants.items():
            if not variant.benchmark_results:
                continue
                
            analysis = {
                "name": variant.name,
                "type": variant.variant_type,
                "compression_ratio": variant.get_compression_ratio(self.baseline_info["model_size_mb"]),
                "speedup_ratio": variant.get_speedup_ratio(baseline.benchmark_results.inference_speed) if baseline.benchmark_results else None,
                "accuracy_retention": variant.get_accuracy_retention(baseline.benchmark_results.token_accuracy) if baseline.benchmark_results else None,
                "model_size_mb": variant.model_size_mb,
                "inference_speed": variant.benchmark_results.inference_speed,
                "token_accuracy": variant.benchmark_results.token_accuracy,
                "peak_memory_mb": variant.benchmark_results.peak_memory_mb
            }
            
            recommendations["variant_analysis"][variant_name] = analysis
            
        # Generate scenario-based recommendations
        scenarios = {
            "high_accuracy": {
                "description": "Maximum accuracy, resource usage not a concern",
                "priority": ["token_accuracy"],
                "recommendation": None
            },
            "balanced": {
                "description": "Good balance of accuracy, speed, and size",
                "priority": ["efficiency_score"],
                "recommendation": None
            },
            "mobile_edge": {
                "description": "Mobile/edge deployment, minimize size and memory",
                "priority": ["compression_ratio", "peak_memory_mb"],
                "recommendation": None
            },
            "real_time": {
                "description": "Real-time applications, maximize speed",
                "priority": ["inference_speed"],
                "recommendation": None
            },
            "batch_processing": {
                "description": "Batch processing, balance throughput and accuracy",
                "priority": ["inference_speed", "token_accuracy"],
                "recommendation": None
            }
        }
        
        # Find best variant for each scenario
        for scenario_name, scenario in scenarios.items():
            best_variant = None
            best_score = -float('inf')
            
            for variant_name, analysis in recommendations["variant_analysis"].items():
                # Calculate scenario-specific score
                score = 0
                
                if scenario_name == "high_accuracy":
                    score = analysis["token_accuracy"]
                elif scenario_name == "balanced":
                    # Efficiency score: accuracy * speed / size
                    score = (analysis["token_accuracy"] * 
                           analysis["inference_speed"] / 
                           max(analysis["model_size_mb"], 1))
                elif scenario_name == "mobile_edge":
                    # Favor high compression and low memory
                    score = (analysis["compression_ratio"] * 10 / 
                           max(analysis["peak_memory_mb"], 1))
                elif scenario_name == "real_time":
                    score = analysis["inference_speed"]
                elif scenario_name == "batch_processing":
                    # Balance of speed and accuracy
                    score = (analysis["inference_speed"] * 
                           analysis["token_accuracy"])
                    
                if score > best_score:
                    best_score = score
                    best_variant = variant_name
                    
            if best_variant:
                scenario["recommendation"] = {
                    "variant": best_variant,
                    "details": recommendations["variant_analysis"][best_variant],
                    "score": best_score
                }
                
        recommendations["scenarios"] = scenarios
        
        # Generate summary insights
        all_variants = list(recommendations["variant_analysis"].values())
        
        recommendations["summary"] = {
            "total_variants": len(all_variants),
            "best_compression": max(all_variants, key=lambda x: x["compression_ratio"])["name"],
            "fastest_inference": max(all_variants, key=lambda x: x["inference_speed"])["name"],
            "highest_accuracy": max(all_variants, key=lambda x: x["token_accuracy"])["name"],
            "most_memory_efficient": min(all_variants, key=lambda x: x["peak_memory_mb"])["name"],
            "average_compression": sum(x["compression_ratio"] for x in all_variants) / len(all_variants),
            "average_speedup": sum(x["speedup_ratio"] for x in all_variants) / len(all_variants),
            "average_accuracy_retention": sum(x["accuracy_retention"] for x in all_variants) / len(all_variants)
        }
        
        return recommendations
        
    def save_all_artifacts(self) -> None:
        """
        Save all variants and results to disk.
        
        This method saves:
        - Individual variant metadata
        - Deployment recommendations
        - Complete packaging report
        """
        logger.info("Saving all packaging artifacts...")
        
        # Save individual variant metadata
        variants_metadata = {}
        for variant_name, variant in self.variants.items():
            variant_metadata = variant.to_dict(self.baseline_info, baseline.benchmark_results)
            variants_metadata[variant_name] = variant_metadata
            
            # Save individual variant file
            variant_file = os.path.join(
                results_dir, 
                f"{variant_name}_metadata.json"
            )
            with open(variant_file, 'w') as f:
                json.dump(variant_metadata, f, indent=2)
                
        # Save combined variants metadata
        combined_metadata_file = os.path.join(
            results_dir, 
            "all_variants_metadata.json"
        )
        with open(combined_metadata_file, 'w') as f:
            json.dump(variants_metadata, f, indent=2)
            
        # Generate and save deployment recommendations
        recommendations = self.generate_deployment_recommendations()
        recommendations_file = os.path.join(
            results_dir,
            "deployment_recommendations.json"
        )
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
            
        # Create packaging report
        packaging_report = {
            "configuration": {
                "wandb_project": self.config.wandb.project,
                "source_run_id": self.config.wandb.source_run_id,
                "variants_created": list(self.variants.keys()),
                "distillation_enabled": self.config.optimization.methods.create_distilled,
                "quantization_enabled": self.config.optimization.methods.create_quantized,
                "combined_enabled": self.config.optimization.methods.create_combined
            },
            "variants_summary": variants_metadata,
            "deployment_recommendations": recommendations,
            "artifacts_locations": {
                "models_dir": self.config.output.output_dir,
                "results_dir": results_dir,
                "benchmarks": os.path.join(results_dir, "model_comparison.json"),
                "recommendations": recommendations_file
            }
        }
        
        # Save complete packaging report
        report_file = os.path.join(
            results_dir,
            "packaging_report.json"
        )
        with open(report_file, 'w') as f:
            json.dump(packaging_report, f, indent=2)
            
        logger.info(f"All artifacts saved to: {results_dir}")
        logger.info(f"Packaging report: {report_file}")
        
    def setup_data_preprocessing(self, main_config_path: str = "configs/config.yml"):
        """
        Set up data preprocessing using existing infrastructure.
        
        Args:
            main_config_path: Path to main training configuration
        """
        logger.info("Setting up data preprocessing...")
        
        # Load main training configuration for data settings
        main_config = load_config(main_config_path)
        
        # Use existing setup_data_and_tokenizer function from data processing
        raw_text, self.tokenizer = setup_data_and_tokenizer(main_config)
        
        # Create preprocessing object with the same settings as training
        self.preprocessor = PreprocessingTraining(
            text=raw_text,
            tokenizer=self.tokenizer,
            batch_size=getattr(main_config, 'BATCH_SIZE', 32),
            time_steps=getattr(main_config, 'CONTEXT_WINDOW', 64)
        )
        
        logger.info(f"Data preprocessing completed | Dataset: {main_config.DATASET_NAME} | Vocab: {self.tokenizer.vocab_size:,} | Tokens - Train: {len(self.preprocessor.train_tokens):,}, Val: {len(self.preprocessor.val_tokens):,}, Test: {len(self.preprocessor.test_tokens):,}")
    
    def get_data_loaders(self):
        """
        Get data loader functions for train/val/test splits.
        
        Returns:
            Tuple of (train_loader_fn, val_loader_fn, test_loader_fn)
        """
        if self.preprocessor is None:
            raise ValueError("Data preprocessing not set up. Call setup_data_preprocessing() first.")
        
        def train_loader():
            """Generator function that yields training batches."""
            while True:
                yield self.preprocessor.get_batch("train")
        
        def val_loader():
            """Generator function that yields validation batches.""" 
            while True:
                yield self.preprocessor.get_batch("val")
        
        def test_loader():
            """Generator function that yields test batches for evaluation."""
            while True:
                yield self.preprocessor.get_batch("test")
        
        return train_loader, val_loader, test_loader
    
    def run_complete_pipeline(self, main_config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete packaging pipeline with internal data setup.
        
        This is the main entry point that handles:
        1. Data preprocessing setup
        2. Model loading and variant creation  
        3. Benchmarking and comparison
        4. Results saving and reporting
        
        Args:
            main_config_path: Path to main training configuration
            
        Returns:
            Dictionary containing complete pipeline results
        """
        logger.info("Starting complete packaging pipeline...")
        
        # Use base model config path from packaging config if not provided
        if main_config_path is None:
            main_config_path = self.config.base_model.config_path
        
        try:
            # 1. Set up data preprocessing
            logger.info("Step 1: Setting up data preprocessing...")
            try:
                self.setup_data_preprocessing(main_config_path)
                train_loader, val_loader, test_loader = self.get_data_loaders()
                logger.info("✓ Data preprocessing completed successfully")
            except FileNotFoundError as e:
                logger.error(f"Data setup failed - configuration file not found: {str(e)}")
                raise FileNotFoundError(f"Data preprocessing setup failed: {str(e)}") from e
            except Exception as e:
                logger.error(f"Data preprocessing setup failed: {str(e)}")
                raise RuntimeError(f"Data preprocessing failed: {str(e)}") from e
            
            # 2. Load main model configuration for architecture details
            logger.info("Step 2: Loading model configuration...")
            try:
                main_config = load_config(main_config_path)
                model_config = {
                    "vocab_size": self.tokenizer.vocab_size,
                    "channel_dim": main_config.CHANNEL_DIM,
                    "context_window": main_config.CONTEXT_WINDOW,
                    "num_heads": main_config.NUM_HEADS,
                    "num_layers": main_config.NUM_LAYERS,
                    "dropout_rate": main_config.DROPOUT_RATE
                }
                
                # Calculate student architectures based on base model
                self._calculate_student_architectures(model_config)
                
                logger.info("✓ Model configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model configuration: {str(e)}")
                raise ValueError(f"Model configuration loading failed: {str(e)}") from e
            
            # 3. Load baseline model
            logger.info("Step 3: Loading baseline model from W&B...")
            try:
                baseline_model = self.load_baseline_model(model_config)
                # Initialize base_model reference in manager
                self.base_model = baseline_model
                logger.info(f"✓ Baseline model loaded: {self.baseline_info['parameter_count']:,} parameters")
            except ConnectionError as e:
                logger.error(f"Failed to connect to W&B for model loading: {str(e)}")
                raise ConnectionError(f"W&B connection failed: {str(e)}") from e
            except FileNotFoundError as e:
                logger.error(f"Baseline model checkpoint not found: {str(e)}")
                raise FileNotFoundError(f"Model checkpoint not found: {str(e)}") from e
            except Exception as e:
                logger.error(f"Baseline model loading failed: {str(e)}")
                raise RuntimeError(f"Baseline model loading failed: {str(e)}") from e
            
            # 4. Create optimized variants
            logger.info("Step 4: Creating optimized model variants...")
            
            # Knowledge distillation
            if self.config.optimization.methods.create_distilled:
                logger.info("Creating distilled variants...")
                try:
                    distilled_variants = self.create_distilled_variants(
                        train_loader, val_loader
                    )
                    logger.info(f"✓ Created {len(distilled_variants)} distilled variants")
                except MemoryError as e:
                    logger.error(f"Out of memory during distillation: {str(e)}")
                    raise MemoryError(f"Distillation failed due to memory: {str(e)}") from e
                except Exception as e:
                    logger.error(f"Distillation failed: {str(e)}")
                    raise RuntimeError(f"Distillation process failed: {str(e)}") from e
            else:
                logger.info("Distillation disabled in configuration")
            
            # Quantization
            if self.config.optimization.methods.create_quantized:
                logger.info("Creating quantized variants...")
                try:
                    quantized_variants = self.create_quantized_variants(
                        train_loader, val_loader
                    )
                    logger.info(f"✓ Created {len(quantized_variants)} quantized variants")
                except RuntimeError as e:
                    logger.error(f"Quantization runtime error: {str(e)}")
                    logger.error("This may indicate CUDA/device compatibility issues")
                    raise RuntimeError(f"Quantization failed: {str(e)}") from e
                except Exception as e:
                    logger.error(f"Quantization failed: {str(e)}")
                    raise RuntimeError(f"Quantization process failed: {str(e)}") from e
            else:
                logger.info("Quantization disabled in configuration")
            
            # Combined (distilled + quantized)
            if self.config.optimization.methods.create_combined:
                logger.info("Creating combined variants...")
                try:
                    combined_variants = self.create_combined_variants(
                        train_loader, val_loader
                    )
                    logger.info(f"✓ Created {len(combined_variants)} combined variants")
                except Exception as e:
                    logger.error(f"Combined optimization failed: {str(e)}")
                    raise RuntimeError(f"Combined optimization process failed: {str(e)}") from e
            else:
                logger.info("Combined optimization disabled in configuration")
            
            # 5. Comprehensive benchmarking
            logger.info("Step 5: Running comprehensive benchmarking...")
            try:
                benchmark_results = self.benchmark_all_variants(test_loader)
                logger.info(f"✓ Benchmarking completed for {len(benchmark_results)} variants")
            except MemoryError as e:
                logger.error(f"Out of memory during benchmarking: {str(e)}")
                logger.warning("Continuing with partial results...")
                benchmark_results = {}
            except Exception as e:
                logger.error(f"Benchmarking failed: {str(e)}")
                logger.warning("Continuing without benchmark results...")
                benchmark_results = {}
            
            # 6. Save results and generate reports
            logger.info("Step 6: Saving results and generating reports...")
            try:
                final_results = self.save_results_and_generate_reports()
                logger.info("✓ Results saved and reports generated successfully")
            except OSError as e:
                logger.error(f"File system error during results saving: {str(e)}")
                raise OSError(f"Results saving failed: {str(e)}") from e
            except Exception as e:
                logger.error(f"Results saving failed: {str(e)}")
                raise RuntimeError(f"Results saving failed: {str(e)}") from e
            
            # 7. Add benchmarking results and performance summary to final output
            try:
                final_results['benchmark_results'] = benchmark_results
                final_results['variants_created'] = len(self.variants)
                final_results['baseline_parameters'] = self.baseline_info['parameter_count']
                final_results['status'] = 'success'
                
                # Log summary with key metrics
                summary_msg = f"PIPELINE SUCCESS | Variants: {len(self.variants)} | Baseline params: {final_results['baseline_parameters']:,}"
                if len(self.variants) > 0:
                    compression_ratios = [variant.get_compression_ratio(self.baseline_info["model_size_mb"]) 
                                        for variant in self.variants.values()]
                    if compression_ratios:
                        best_compression = max(compression_ratios)
                        summary_msg += f" | Best compression: {best_compression:.1f}x"
                summary_msg += f" | Location: {self.config.output.output_dir}"
                
                logger.info(summary_msg)
                
                return final_results
                
            except Exception as e:
                logger.error(f"Error finalizing results: {str(e)}")
                # Return basic results even if finalization fails
                return {
                    'status': 'partial_success',
                    'variants_created': len(self.variants),
                    'baseline_parameters': self.baseline_info['parameter_count'],
                    'error_message': f"Finalization error: {str(e)}"
                }
                
        except KeyboardInterrupt:
            logger.error("Pipeline execution interrupted by user")
            raise KeyboardInterrupt("Pipeline execution was interrupted")
        except Exception as e:
            # Final catch-all with context-specific error messages
            logger.error(f"PIPELINE FAILED | Stage: {self._get_current_stage()} | Error: {type(e).__name__}: {str(e)}")
            
            # Return error information for analysis
            error_results = {
                'status': 'failed',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'variants_created': len(self.variants) if hasattr(self, 'variants') else 0,
                'baseline_parameters': self.baseline_info['parameter_count'],
                'pipeline_stage': self._get_current_stage()
            }
            
            # Log the error but don't re-raise - let caller handle
            logger.error(f"Returning error results: {error_results}")
            raise
    
    def _get_current_stage(self) -> str:
        """
        Determine the current pipeline stage for error reporting.
        
        Returns:
            String describing the current stage
        """
        if not hasattr(self, 'variants'):
            return "initialization"
        elif len(self.variants) == 0:
            return "baseline_loading"
        elif len(self.variants) == 1:
            return "variant_creation"
        else:
            return "benchmarking_or_results" 
    
    def save_results_and_generate_reports(self) -> Dict[str, Any]:
        """
        Save all results and generate comprehensive reports.
        
        Returns:
            Dictionary containing all results and metadata
        """
        logger.info("Saving results and generating reports...")
        
        results_dir = os.path.join(self.config.output.output_dir, self.config.output.results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        # Compile comprehensive results
        results = {
            "pipeline_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "wandb_project": self.config.wandb.project,
                "source_run_id": self.config.wandb.source_run_id,
                "config": {
                    "distillation_enabled": self.config.optimization.methods.create_distilled,
                    "quantization_enabled": self.config.optimization.methods.create_quantized,
                    "combined_enabled": self.config.optimization.methods.create_combined
                }
            },
            "output_info": {
                "models_dir": self.config.output.output_dir,
                "results_dir": results_dir
            },
            "variants": {}
        }
        
        # Add variant information with comprehensive metadata for future use
        for variant_name, variant in self.variants.items():
            # Use the updated to_dict method that includes performance ratios
            variant_dict = variant.to_dict(self.baseline_info, None)  # No baseline benchmark results available here
            
            # Add additional metadata for model usage
            variant_dict["usage_info"] = {
                "model_class": "TransformerModel",
                "model_module": "src.models.transformer",
                "tokenizer_info": {
                    "vocab_size": variant.model.vocab_size if hasattr(variant.model, 'vocab_size') else "unknown",
                    "context_window": getattr(variant.model, 'context_window', "unknown")
                },
                "deployment_recommendations": self._get_deployment_recommendations(variant),
                "performance_profile": self._get_performance_profile(variant)
            }
            
            results["variants"][variant_name] = variant_dict
        
        # Save results to JSON
        results_file = os.path.join(results_dir, "packaging_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Generate summary report
        self._generate_summary_report(results_dir, results)
        
        return results
    
    def _generate_summary_report(self, results_dir: str, results: Dict[str, Any]) -> None:
        """
        Generate a human-readable summary report.
        
        Args:
            results_dir: Directory to save the report
            results: Results dictionary
        """
        report_file = os.path.join(results_dir, "packaging_summary.txt")
        
        with open(report_file, 'w') as f:
            f.write("MODEL PACKAGING SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {results['pipeline_info']['timestamp']}\n")
            f.write(f"Source Run: {results['pipeline_info']['source_run_id']}\n")
            f.write(f"W&B Project: {results['pipeline_info']['wandb_project']}\n\n")
            
            f.write("VARIANTS CREATED:\n")
            f.write("-" * 20 + "\n")
            
            for variant_name, variant_info in results["variants"].items():
                f.write(f"\n{variant_name.upper()}:\n")
                f.write(f"  Type: {variant_info['variant_type']}\n")
                f.write(f"  Parameters: {variant_info['parameter_count']:,}\n")
                f.write(f"  Size: {variant_info['model_size_mb']:.1f} MB\n")
                
                # Performance ratios (key metrics for developers)
                if 'performance_ratios' in variant_info:
                    ratios = variant_info['performance_ratios']
                    f.write(f"  Compression Ratio: {ratios['compression_ratio']:.2f}x\n")
                    if ratios['speedup_ratio'] is not None:
                        f.write(f"  Speedup Ratio: {ratios['speedup_ratio']:.2f}x\n")
                    if ratios['accuracy_retention'] is not None:
                        f.write(f"  Accuracy Retention: {ratios['accuracy_retention']:.1%}\n")
                
                f.write(f"  Checkpoint: {variant_info['checkpoint_path']}\n")
                f.write(f"  W&B Run: {variant_info['run_id']}\n")
                
                if 'benchmark_results' in variant_info:
                    bench = variant_info['benchmark_results']
                    f.write(f"  Inference Speed: {bench.get('inference_speed', 'N/A')}\n")
                    f.write(f"  Memory Usage: {bench.get('memory_usage_mb', 'N/A')} MB\n")
            
            # Add performance comparison summary
            f.write(f"\nPERFORMANCE COMPARISON:\n")
            f.write("-" * 25 + "\n")
            
            # Create a comparison table
            f.write(f"{'Variant':<20} {'Compression':<12} {'Speedup':<10} {'Accuracy':<12} {'Use Case':<30}\n")
            f.write("-" * 84 + "\n")
            
            for variant_name, variant_info in results["variants"].items():
                name = variant_info['variant_type'].capitalize()[:19]
                
                if 'performance_ratios' in variant_info:
                    ratios = variant_info['performance_ratios']
                    compression = f"{ratios['compression_ratio']:.1f}x"
                    speedup = f"{ratios['speedup_ratio']:.1f}x" if ratios['speedup_ratio'] is not None else "N/A"
                    accuracy = f"{ratios['accuracy_retention']:.0%}" if ratios['accuracy_retention'] is not None else "N/A"
                else:
                    compression = speedup = accuracy = "N/A"
                
                # Get use case from deployment recommendations
                use_case = "Standard deployment"
                if 'usage_info' in variant_info and 'deployment_recommendations' in variant_info['usage_info']:
                    use_case = variant_info['usage_info']['deployment_recommendations'].get('deployment_target', 'Standard deployment')[:29]
                
                f.write(f"{name:<20} {compression:<12} {speedup:<10} {accuracy:<12} {use_case:<30}\n")
            
            f.write(f"\nResults saved to: {results_dir}\n")
        
        logger.info(f"Summary report saved to: {report_file}")
    
    def _get_deployment_recommendations(self, variant: ModelVariant) -> Dict[str, str]:
        """
        Get deployment recommendations for a model variant.
        
        Args:
            variant: Model variant to analyze
            
        Returns:
            Dictionary with deployment recommendations
        """
        recommendations = {}
        
        if variant.variant_type == "baseline":
            recommendations = {
                "use_case": "High-accuracy scenarios where model size is not a constraint",
                "deployment_target": "Cloud servers, high-end workstations",
                "optimization_level": "None - original model",
                "trade_offs": "Highest accuracy, largest size, slowest inference"
            }
        elif variant.variant_type == "distilled":
            recommendations = {
                "use_case": "Balanced accuracy and efficiency for most applications",
                "deployment_target": "Standard servers, mid-range devices",
                "optimization_level": "Knowledge distillation - reduced model size",
                "trade_offs": "Good accuracy retention, moderate size reduction, faster inference"
            }
        elif variant.variant_type == "quantized":
            recommendations = {
                "use_case": "Fast inference with minimal accuracy loss",
                "deployment_target": "Edge devices, mobile applications, real-time systems",
                "optimization_level": "INT8 quantization - 4x size reduction",
                "trade_offs": "Minimal accuracy loss, significant size reduction, fastest inference"
            }
        elif variant.variant_type == "combined":
            recommendations = {
                "use_case": "Extremely resource-constrained environments",
                "deployment_target": "IoT devices, embedded systems, mobile apps",
                "optimization_level": "Distillation + Quantization - maximum compression",
                "trade_offs": "Moderate accuracy loss, maximum size reduction, very fast inference"
            }
        
        return recommendations
    
    def _get_performance_profile(self, variant: ModelVariant) -> Dict[str, str]:
        """
        Get performance profile for a model variant.
        
        Args:
            variant: Model variant to analyze
            
        Returns:
            Dictionary with performance characteristics
        """
        profile = {
            "model_type": variant.variant_type,
            "parameter_count": f"{variant.parameter_count:,}",
            "model_size_mb": f"{variant.model_size_mb:.1f}",
            "optimization_techniques": []
        }
        
        if variant.variant_type == "distilled":
            profile["optimization_techniques"].append("Knowledge Distillation")
        elif variant.variant_type == "quantized":
            profile["optimization_techniques"].append("INT8 Quantization")
        elif variant.variant_type == "combined":
            profile["optimization_techniques"].extend(["Knowledge Distillation", "INT8 Quantization"])
        
        if variant.benchmark_results:
            profile.update({
                "inference_speed": f"{variant.benchmark_results.inference_speed:.2f} tokens/sec" if hasattr(variant.benchmark_results, 'inference_speed') else "Not measured",
                "memory_usage": f"{variant.benchmark_results.memory_usage_mb:.1f} MB" if hasattr(variant.benchmark_results, 'memory_usage_mb') else "Not measured",
                "accuracy_metrics": "Available in benchmark_results" if variant.benchmark_results else "Not available"
            })
        
        return profile

    def upload_to_wandb_with_timeout(self, model_path: str, run_name: str, timeout: int = 300) -> bool:
        """
        Upload model to W&B with timeout protection.
        
        Args:
            model_path: Path to model file
            run_name: Name for the W&B run
            timeout: Timeout in seconds
            
        Returns:
            True if upload successful, False otherwise
        """
        def upload_worker():
            try:
                wandb.init(
                    project=self.config.wandb.project,
                    name=run_name,
                    job_type="model_upload"
                )
                artifact = wandb.Artifact(run_name, type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                wandb.finish()
                return True
            except Exception as e:
                logger.error(f"W&B upload failed: {str(e)}")
                return False
        
        # Upload in separate thread with timeout
        upload_thread = threading.Thread(target=upload_worker)
        upload_thread.start()
        upload_thread.join(timeout=timeout)
        
        if upload_thread.is_alive():
            logger.warning(f"W&B upload timed out after {timeout} seconds")
            return False
        
        return True

    def cleanup_training_artifacts(self, save_dir: str, keep_best: bool = True) -> None:
        """
        Clean up temporary training artifacts.
        
        Args:
            save_dir: Directory to clean
            keep_best: Whether to keep best model checkpoints
        """
        # Patterns to remove
        cleanup_patterns = [
            "*.tmp", "*.temp", "*_intermediate_*.pt", 
            "*_epoch_*.pt", "optimizer_*.pt"
        ]
        
        for pattern in cleanup_patterns:
            for file_path in glob.glob(os.path.join(save_dir, pattern)):
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {str(e)}")
        
        # Remove empty directories
        for root, dirs, files in os.walk(save_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logger.debug(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Could not remove directory {dir_path}: {str(e)}")

    def _log_distillation_metrics_to_wandb(self, student_name: str, training_results: Dict[str, Any], 
                                          optimization_type: str = "knowledge_distillation") -> str:
        """
        Log distillation training metrics to W&B.
        
        Args:
            student_name: Name of the student model
            training_results: Results from distillation training
            optimization_type: Type of optimization ("knowledge_distillation" or "quantization_aware_distillation")
            
        Returns:
            W&B run ID for the logged experiment
        """
        import time
        
        # Create run name based on optimization type
        timestamp = int(time.time())
        if optimization_type == "quantization_aware_distillation":
            run_name = f"qad_{student_name}_{timestamp}"
            tags = ["quantization_aware_distillation", "qad", student_name]
        else:
            run_name = f"distill_{student_name}_{timestamp}"
            tags = ["distillation", "knowledge_transfer", student_name]
        
        # Initialize W&B run
        wandb.init(
            project=self.config.wandb.project,
            name=run_name,
            tags=tags,
            config={
                "optimization_type": optimization_type,
                "student_architecture": self.calculated_student_architectures.get(student_name, {}),
                "distillation_config": {
                    "temperature": self.config.optimization.distillation.temperature,
                    "alpha": self.config.optimization.distillation.alpha,
                    "learning_rate": self.config.optimization.distillation.learning_rate
                },
                "quantization_aware": optimization_type == "quantization_aware_distillation"
            }
        )
        
        try:
            # Log all training metrics
            if "training_metrics" in training_results:
                for metrics in training_results["training_metrics"]:
                    wandb.log(metrics)
            
            # Log final results
            wandb.log({
                "final_step": training_results.get("final_step", 0),
                "best_val_loss": training_results.get("best_val_loss", 0),
                "compression_ratio": training_results.get("compression_ratio", 1.0),
                "training_completed": True
            })
            
            # Get run ID for tracking
            run_id = wandb.run.id
            
            logger.info(f"Logged {optimization_type} metrics for {student_name} to W&B run: {run_id}")
            
            return run_id
            
        finally:
            wandb.finish() 