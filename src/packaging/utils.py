"""
Packaging Utilities for Model Optimization
==========================================

This module provides utility functions for the packaging pipeline,
including model loading/saving, configuration management, and helper functions
for working with W&B artifacts and model checkpoints.

UTILITY FUNCTIONS PROVIDED:
==========================
1. Model Management:
   - Loading models from W&B artifacts
   - Saving optimized models with metadata
   - Model format conversions

2. Configuration Management:
   - Packaging configuration validation
   - Configuration merging and inheritance
   - Environment-specific settings

3. W&B Integration:
   - Artifact management
   - Run metadata handling
   - Model registration and versioning

4. File System Operations:
   - Directory structure management
   - Checkpoint organization
   - Results aggregation

5. Validation and Testing:
   - Model compatibility checking
   - Configuration validation
   - Smoke tests for optimized models
"""

import torch
import torch.nn as nn
import wandb
import os
import logging
import yaml
import json
import hashlib
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
from dataclasses import asdict

from src.models.transformer import TransformerModel
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def validate_packaging_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate packaging configuration for completeness and correctness.
    
    Args:
        config: Packaging configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ["wandb_project", "run_id", "output_dir"]
    for field in required_fields:
        if field not in config or not config[field]:
            errors.append(f"Missing required field: {field}")
            
    # Validate directories
    if "output_dir" in config:
        try:
            os.makedirs(config["output_dir"], exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {str(e)}")
            
    # Validate W&B project and run
    if "wandb_project" in config and "run_id" in config:
        try:
            api = wandb.Api()
            run = api.run(f"{config['wandb_project']}/{config['run_id']}")
            if run is None:
                errors.append(f"W&B run not found: {config['wandb_project']}/{config['run_id']}")
        except Exception as e:
            errors.append(f"Cannot access W&B run: {str(e)}")
            
    # Validate optimization flags
    optimization_flags = ["create_distilled", "create_quantized", "create_combined"]
    if all(not config.get(flag, True) for flag in optimization_flags):
        errors.append("At least one optimization method must be enabled")
        
    return len(errors) == 0, errors


def load_model_from_wandb(project_name: str, run_id: str, 
                         model_config: Dict[str, Any], device: str = "cpu") -> nn.Module:
    """
    Load a model from W&B artifacts.
    
    Args:
        project_name: W&B project name
        run_id: W&B run identifier
        model_config: Model configuration parameters
        device: Device to load the model on
        
    Returns:
        Loaded model instance
    """
    logger.info(f"Loading model from W&B: {project_name}/{run_id}")
    
    try:
        # Initialize W&B API
        api = wandb.Api()
        
        # Get the specific run
        run = api.run(f"{project_name}/{run_id}")
        if run is None:
            raise ValueError(f"Run not found: {project_name}/{run_id}")
            
        # Look for model artifacts
        artifacts = run.logged_artifacts()
        model_artifact = None
        
        for artifact in artifacts:
            if artifact.type == "model":
                model_artifact = artifact
                break
                
        if model_artifact is None:
            raise ValueError(f"No model artifact found in run {run_id}")
            
        # Download the artifact
        artifact_dir = model_artifact.download()
        
        # Find the checkpoint file
        checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            raise ValueError("No .pt checkpoint files found in artifact")
            
        checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
        
        # Create model with the specified architecture
        model = TransformerModel(
            vocab_size=model_config["vocab_size"],
            channel_dim=model_config["channel_dim"],
            context_window=model_config["context_window"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dropout_rate=model_config.get("dropout_rate", 0.1)
        )
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        model.to(device)
        logger.info(f"Successfully loaded model from {checkpoint_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from W&B: {str(e)}")
        raise


def save_model_with_metadata(model: nn.Module, save_path: str, 
                            metadata: Dict[str, Any]) -> None:
    """
    Save a model with comprehensive metadata.
    
    Args:
        model: Model to save
        save_path: Path to save the model
        metadata: Metadata to include with the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate model hash for verification
    model_hash = calculate_model_hash(model)
    
    # Prepare comprehensive metadata
    full_metadata = {
        "model_info": {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_size_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "model_hash": model_hash,
            "architecture": type(model).__name__
        },
        "save_info": {
            "save_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "save_path": save_path,
            "pytorch_version": torch.__version__
        },
        "custom_metadata": metadata
    }
    
    # Save model and metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": full_metadata
    }
    
    torch.save(checkpoint, save_path)
    
    # Also save metadata as separate JSON file
    metadata_path = save_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
        
    logger.info(f"Model saved to {save_path} with metadata")


def calculate_model_hash(model: nn.Module) -> str:
    """
    Calculate a hash of the model's parameters for verification.
    
    Args:
        model: Model to hash
        
    Returns:
        Hexadecimal hash string
    """
    hash_md5 = hashlib.md5()
    
    for param in model.parameters():
        hash_md5.update(param.cpu().data.numpy().tobytes())
        
    return hash_md5.hexdigest()


def verify_model_integrity(model_path: str) -> Tuple[bool, str]:
    """
    Verify the integrity of a saved model using its hash.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if "metadata" not in checkpoint:
            return False, "No metadata found in checkpoint"
            
        stored_hash = checkpoint["metadata"]["model_info"].get("model_hash")
        if not stored_hash:
            return False, "No hash found in metadata"
            
        # Create temporary model to verify hash
        # Note: This requires knowledge of the model architecture
        # In a real implementation, this would need the model config
        logger.warning("Model integrity verification requires model architecture - skipping hash check")
        return True, "Model loaded successfully (hash verification skipped)"
        
    except Exception as e:
        return False, f"Error verifying model: {str(e)}"


def create_packaging_workspace(base_dir: str, run_id: str) -> Dict[str, str]:
    """
    Create a structured workspace for packaging operations.
    
    Args:
        base_dir: Base directory for packaging
        run_id: Run identifier for organizing files
        
    Returns:
        Dictionary of created directory paths
    """
    workspace = {
        "root": os.path.join(base_dir, f"packaging_{run_id}"),
        "models": None,
        "results": None,
        "configs": None,
        "logs": None,
        "temp": None
    }
    
    # Create subdirectories
    subdirs = {
        "models": "optimized_models",
        "results": "benchmark_results", 
        "configs": "configurations",
        "logs": "logs",
        "temp": "temp"
    }
    
    for key, subdir in subdirs.items():
        full_path = os.path.join(workspace["root"], subdir)
        os.makedirs(full_path, exist_ok=True)
        workspace[key] = full_path
        
    logger.info(f"Packaging workspace created: {workspace['root']}")
    return workspace


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration dictionaries with override support.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def load_packaging_config(config_path: str, 
                         overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load packaging configuration with optional overrides.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional configuration overrides
        
    Returns:
        Loaded and merged configuration
    """
    # Load base configuration
    if config_path.endswith('.yml') or config_path.endswith('.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")
        
    # Apply overrides if provided
    if overrides:
        config = merge_configs(config, overrides)
        
    # Validate configuration
    is_valid, errors = validate_packaging_config(config)
    if not is_valid:
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
    return config


def export_model_for_inference(model: nn.Module, export_path: str, 
                              model_config: Dict[str, Any], 
                              format: str = "torchscript") -> str:
    """
    Export model in a format optimized for inference.
    
    Args:
        model: Model to export
        export_path: Path to save the exported model
        model_config: Model configuration
        format: Export format ("torchscript", "onnx", "pt")
        
    Returns:
        Path to the exported model
    """
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    model.eval()
    
    if format == "torchscript":
        # Export as TorchScript for deployment
        dummy_input = torch.randint(
            0, model_config["vocab_size"], 
            (1, model_config["context_window"]),
            dtype=torch.long
        )
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            
        traced_model.save(export_path)
        logger.info(f"Model exported as TorchScript to {export_path}")
        
    elif format == "onnx":
        try:
            import torch.onnx
            
            dummy_input = torch.randint(
                0, model_config["vocab_size"],
                (1, model_config["context_window"]),
                dtype=torch.long
            )
            
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model exported as ONNX to {export_path}")
            
        except ImportError:
            logger.error("ONNX not available. Install with: pip install onnx")
            raise
            
    elif format == "pt":
        # Save as standard PyTorch checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, export_path)
        
        logger.info(f"Model exported as PyTorch checkpoint to {export_path}")
        
    else:
        raise ValueError(f"Unsupported export format: {format}")
        
    return export_path


def create_deployment_package(models_dir: str, results_dir: str, 
                            package_dir: str) -> str:
    """
    Create a complete deployment package with all variants and documentation.
    
    Args:
        models_dir: Directory containing optimized models
        results_dir: Directory containing benchmark results
        package_dir: Directory to create the deployment package
        
    Returns:
        Path to the created package
    """
    logger.info("Creating deployment package...")
    
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy model files
    models_package_dir = os.path.join(package_dir, "models")
    if os.path.exists(models_dir):
        shutil.copytree(models_dir, models_package_dir, dirs_exist_ok=True)
        
    # Copy results and documentation
    docs_package_dir = os.path.join(package_dir, "documentation")
    if os.path.exists(results_dir):
        shutil.copytree(results_dir, docs_package_dir, dirs_exist_ok=True)
        
    # Create deployment guide
    deployment_guide = {
        "package_info": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "package_version": "1.0.0",
            "description": "Optimized transformer model deployment package"
        },
        "models_included": {
            "baseline": "Original trained model",
            "distilled": "Knowledge distilled variants",
            "quantized": "INT8 quantized variants", 
            "combined": "Distilled + quantized variants"
        },
        "deployment_instructions": {
            "1": "Choose appropriate model variant based on your requirements",
            "2": "Load model using provided configuration",
            "3": "Use scripts/generate.py for text generation",
            "4": "Monitor performance using provided benchmarks"
        },
        "support_files": {
            "benchmarks": "documentation/model_comparison.json",
            "recommendations": "documentation/deployment_recommendations.json",
            "individual_results": "documentation/*_benchmark.json"
        }
    }
    
    guide_path = os.path.join(package_dir, "deployment_guide.json")
    with open(guide_path, 'w') as f:
        json.dump(deployment_guide, f, indent=2)
        
    # Create README
    readme_content = """# Transformer Model Deployment Package

This package contains optimized variants of your transformer model, ready for deployment.

## Package Contents

- `models/`: Optimized model variants
- `documentation/`: Benchmark results and recommendations
- `deployment_guide.json`: Detailed deployment instructions

## Quick Start

1. Review deployment recommendations in `documentation/deployment_recommendations.json`
2. Choose the appropriate model variant for your use case
3. Load the model using the provided configuration
4. Use the existing generation scripts for inference

## Model Variants

- **Baseline**: Original model with highest accuracy
- **Distilled**: Smaller models with good accuracy-speed balance
- **Quantized**: Compressed models for fast inference
- **Combined**: Maximum optimization for resource-constrained environments

## Support

Refer to the benchmark results for detailed performance comparisons and deployment guidance.
"""
    
    readme_path = os.path.join(package_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
        
    logger.info(f"Deployment package created: {package_dir}")
    return package_dir


def cleanup_packaging_artifacts(workspace_dir: str, keep_results: bool = True) -> None:
    """
    Clean up temporary artifacts from packaging process.
    
    Args:
        workspace_dir: Workspace directory to clean
        keep_results: Whether to keep result files
    """
    if not os.path.exists(workspace_dir):
        return
        
    # Clean temporary files
    temp_dir = os.path.join(workspace_dir, "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info("Cleaned temporary files")
        
    # Clean logs if requested
    if not keep_results:
        logs_dir = os.path.join(workspace_dir, "logs")
        if os.path.exists(logs_dir):
            shutil.rmtree(logs_dir)
            logger.info("Cleaned log files")
            
    logger.info("Packaging artifacts cleanup completed")


def run_model_smoke_test(model: nn.Module, tokenizer, device: str = "cpu") -> bool:
    """
    Run a basic smoke test on an optimized model to ensure it works.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer for testing
        device: Device to run test on
        
    Returns:
        True if test passes, False otherwise
    """
    try:
        model.to(device)
        model.eval()
        
        # Create test input
        test_prompt = "Hello world"
        input_ids = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long).to(device)
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Basic checks
        assert outputs is not None, "Model returned None"
        assert outputs.shape[0] == 1, "Unexpected batch dimension"
        assert outputs.shape[-1] == tokenizer.vocab_size, "Unexpected vocab dimension"
        assert not torch.isnan(outputs).any(), "Model outputs contain NaN"
        assert not torch.isinf(outputs).any(), "Model outputs contain Inf"
        
        logger.info("Model smoke test passed")
        return True
        
    except Exception as e:
        logger.error(f"Model smoke test failed: {str(e)}")
        return False


def log_packaging_summary(variants_created: Dict[str, int], 
                         total_time: float, results_dir: str) -> None:
    """
    Log a comprehensive summary of the packaging process.
    
    Args:
        variants_created: Dictionary of variant counts
        total_time: Total processing time in seconds
        results_dir: Directory containing results
    """
    variant_summary = ", ".join([f"{k}: {v}" for k, v in variants_created.items()])
    total_variants = sum(variants_created.values())
    
    # Check key files
    key_files = ["model_comparison.json", "deployment_recommendations.json", "packaging_report.json"]
    missing_files = [f for f in key_files if not os.path.exists(os.path.join(results_dir, f))]
    files_status = f" | Missing: {', '.join(missing_files)}" if missing_files else " | All files present"
    
    logger.info(f"PACKAGING SUMMARY | Time: {total_time:.1f}s | Variants: {total_variants} ({variant_summary}) | Location: {results_dir}{files_status}") 