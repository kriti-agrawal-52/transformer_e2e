#!/usr/bin/env python3
"""
Model Packaging Pipeline Entry Point
====================================

This script provides the main entry point for running the complete model packaging
and optimization pipeline. It orchestrates knowledge distillation, quantization,
benchmarking, and deployment preparation.

WHAT THIS SCRIPT DOES:
=====================
1. Loads packaging configuration from YAML file (following existing patterns)
2. Sets up comprehensive logging system
3. Prepares data loaders for training and evaluation
4. Runs the complete packaging pipeline:
   - Load baseline model from W&B
   - Create distilled variants (knowledge distillation)
   - Create quantized variants (INT8 quantization)
   - Create combined variants (distilled + quantized)
   - Benchmark all variants comprehensively
   - Generate deployment recommendations
   - Save all artifacts and results

USAGE:
======
Basic usage:
    python scripts/package_models.py

With custom config:
    python scripts/package_models.py --config configs/custom_packaging.yml

OUTPUTS:
========
- Optimized model variants in specified output directory
- Comprehensive benchmark results and comparisons
- Deployment recommendations for different scenarios
- Complete packaging report with all metadata
"""

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add the root directory to Python path so we can import from src/
root_dir = Path(__file__).parent.parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import yaml

from src.packaging import ModelVariantsManager
from src.utils.config_loader import load_config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def get_enabled_methods(cfg) -> dict:
    """
    Get enabled optimization methods from configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary with method names as keys and boolean values
    """
    if not hasattr(cfg, 'optimization') or not hasattr(cfg.optimization, 'methods'):
        return {}
        
    methods = cfg.optimization.methods
    return {
        'distillation': getattr(methods, 'create_distilled', False),
        'quantization': getattr(methods, 'create_quantized', False),
        'combined': getattr(methods, 'create_combined', False)
    }


def validate_packaging_params(cfg):
    """
    Validates packaging parameters from config with granular error handling.
    
    Each validation step is checked individually with specific error messages
    and logging, following the patterns from the modeling branch.
    """
    logger.info("Validating packaging configuration parameters...")
    
    try:
        # Validate W&B configuration
        logger.debug("Checking W&B configuration...")
        if not hasattr(cfg, 'wandb'):
            logger.error("Missing 'wandb' section in configuration")
            raise ValueError("Configuration must include a 'wandb' section with project and source_run_id")
            
        try:
            source_run_id = cfg.wandb.source_run_id
            if not source_run_id or not source_run_id.strip():
                logger.error("wandb.source_run_id is empty or contains only whitespace")
                raise ValueError("wandb.source_run_id is required and cannot be empty. Please provide a valid W&B run ID.")
            logger.debug(f"Valid source_run_id: {source_run_id}")
        except AttributeError:
            logger.error("wandb.source_run_id attribute not found in configuration")
            raise ValueError("wandb.source_run_id is required in the configuration")
            
        try:
            wandb_project = cfg.wandb.project
            if not wandb_project or not wandb_project.strip():
                logger.error("wandb.project is empty or contains only whitespace")
                raise ValueError("wandb.project is required and cannot be empty. Please provide a valid W&B project name.")
            logger.debug(f"Valid wandb_project: {wandb_project}")
        except AttributeError:
            logger.error("wandb.project attribute not found in configuration")
            raise ValueError("wandb.project is required in the configuration")
            
    except Exception as e:
        logger.error(f"W&B configuration validation failed: {str(e)}")
        raise
    
    try:
        # Validate base model configuration
        logger.debug("Checking base model configuration...")
        if not hasattr(cfg, 'base_model'):
            logger.error("Missing 'base_model' section in configuration")
            raise ValueError("Configuration must include a 'base_model' section with config_path")
            
        try:
            config_path = cfg.base_model.config_path
            if not config_path or not config_path.strip():
                logger.error("base_model.config_path is empty or contains only whitespace")
                raise ValueError("base_model.config_path is required and cannot be empty. Please provide a valid path to the base model configuration.")
            
            # Check if the config file exists
            if not os.path.exists(config_path):
                logger.error(f"Base model config file not found: {config_path}")
                raise ValueError(f"Base model config file does not exist: {config_path}")
                
            logger.debug(f"Valid base model config path: {config_path}")
        except AttributeError:
            logger.error("base_model.config_path attribute not found in configuration")
            raise ValueError("base_model.config_path is required in the configuration")
            
    except Exception as e:
        logger.error(f"Base model configuration validation failed: {str(e)}")
        raise
    
    try:
        # Validate optimization methods
        logger.debug("Checking optimization methods configuration...")
        if not hasattr(cfg, 'optimization'):
            logger.error("Missing 'optimization' section in configuration")
            raise ValueError("Configuration must include an 'optimization' section")
            
        if not hasattr(cfg.optimization, 'methods'):
            logger.error("Missing 'methods' subsection in optimization configuration")
            raise ValueError("Optimization configuration must include a 'methods' subsection")
            
        # Get enabled methods using helper function
        enabled_methods = get_enabled_methods(cfg)
        
        logger.debug(f"Optimization methods: {enabled_methods}")
        
        # Check if at least one method is enabled
        if not any(enabled_methods.values()):
            logger.error("No optimization methods are enabled")
            raise ValueError(
                "At least one optimization method must be enabled. "
                "Please set one of: create_distilled, create_quantized, or create_combined to true."
            )
        
        # Log enabled methods
        active_methods = [method for method, enabled in enabled_methods.items() if enabled]
        logger.info(f"Enabled optimization methods: {', '.join(active_methods)}")
        
    except Exception as e:
        logger.error(f"Optimization methods validation failed: {str(e)}")
        raise
    
    try:
        # Validate output configuration
        logger.debug("Checking output configuration...")
        if not hasattr(cfg, 'output'):
            logger.error("Missing 'output' section in configuration")
            raise ValueError("Configuration must include an 'output' section")
            
        try:
            output_dir = cfg.output.output_dir
            if not output_dir or not output_dir.strip():
                logger.error("output.output_dir is empty or contains only whitespace")
                raise ValueError("output.output_dir is required and cannot be empty")
            logger.debug(f"Valid output base directory: {output_dir}")
        except AttributeError:
            logger.error("output.output_dir attribute not found in configuration")
            raise ValueError("output.output_dir is required in the configuration")
            
    except Exception as e:
        logger.error(f"Output configuration validation failed: {str(e)}")
        raise
    
    try:
        # Validate compute configuration
        logger.debug("Checking compute configuration...")
        if hasattr(cfg, 'compute') and hasattr(cfg.compute, 'device'):
            device = cfg.compute.device
            valid_devices = ['auto', 'cpu', 'cuda', 'mps']
            if device not in valid_devices:
                logger.warning(f"Unusual device specified: {device}. Valid options: {valid_devices}")
            logger.debug(f"✓ Device setting: {device}")
        else:
            logger.debug("Using default device setting (auto)")
            
    except Exception as e:
        logger.error(f"Compute configuration validation failed: {str(e)}")
        raise
    
    logger.info("All packaging configuration parameters validated successfully")


def setup_logging():
    """
    Set up logging for the packaging pipeline.
    """
    log_file = "packaging_transformer_e2e.log"
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Log startup information
    cuda_info = f" | CUDA: {torch.version.cuda} ({torch.cuda.device_count()} GPUs)" if torch.cuda.is_available() else " | CUDA: Not available"
    logger.info(f"MODEL PACKAGING PIPELINE STARTED | PyTorch: {torch.__version__}{cuda_info} | Log: {log_file}")


def run_packaging_pipeline(packaging_cfg) -> dict:
    """
    Run the complete packaging pipeline using ModelVariantsManager.
    
    Args:
        packaging_cfg: Packaging configuration SimpleNamespace
        
    Returns:
        Dictionary containing pipeline results
    """
    logger.info("Starting complete packaging pipeline...")
    
    start_time = time.time()
    
    try:
        # Initialize the model variants manager
        logger.info("Initializing ModelVariantsManager...")
        try:
            manager = ModelVariantsManager(packaging_cfg)
            logger.info("✓ ModelVariantsManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ModelVariantsManager: {str(e)}")
            logger.error(f"This usually indicates a configuration issue or missing dependencies")
            raise ValueError(f"ModelVariantsManager initialization failed: {str(e)}") from e
        
        # Run the complete pipeline (data setup is handled internally)
        logger.info("Running complete packaging pipeline...")
        try:
            results = manager.run_complete_pipeline()
            logger.info("Pipeline execution completed successfully")
        except FileNotFoundError as e:
            logger.error(f"Required file not found during pipeline execution: {str(e)}")
            logger.error("This usually indicates missing model checkpoints or configuration files")
            raise FileNotFoundError(f"Pipeline failed due to missing file: {str(e)}") from e
        except ConnectionError as e:
            logger.error(f"Network connection error during pipeline execution: {str(e)}")
            logger.error("This usually indicates W&B connection issues or dataset download problems")
            raise ConnectionError(f"Pipeline failed due to connection error: {str(e)}") from e
        except MemoryError as e:
            logger.error(f"Out of memory during pipeline execution: {str(e)}")
            logger.error("Consider reducing batch size or using a smaller model")
            raise MemoryError(f"Pipeline failed due to insufficient memory: {str(e)}") from e
        except RuntimeError as e:
            logger.error(f"Runtime error during pipeline execution: {str(e)}")
            logger.error("This usually indicates CUDA issues or model loading problems")
            raise RuntimeError(f"Pipeline failed with runtime error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during pipeline execution: {str(e)}")
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            raise Exception(f"Pipeline failed with unexpected error: {str(e)}") from e
        
        # Process and log results
        try:
            params_info = f" | Baseline params: {results['baseline_parameters']:,}" if 'baseline_parameters' in results else ""
            logger.info(f"PACKAGING PIPELINE COMPLETED | Variants: {results.get('variants_created', 0)}{params_info}")
            logger.info(f"Results saved to: {packaging_cfg.output.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    except KeyboardInterrupt:
        logger.error("Pipeline execution interrupted by user (Ctrl+C)")
        raise KeyboardInterrupt("Pipeline execution was interrupted by user")
    except SystemExit:
        logger.error("Pipeline execution terminated by system exit")
        raise
    except Exception as e:
        logger.error(f"PACKAGING PIPELINE FAILED: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """
    Main function to run the packaging pipeline, following existing script patterns.
    """
    try:
        # Set up logging first
        setup_logging()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run Model Packaging and Optimization Pipeline.")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/packaging_config.yml",
            help="Path to the packaging configuration YAML file.",
        )
        args = parser.parse_args()
        logger.info(f"Using config file: {args.config}")

        # Load and validate packaging configuration
        logger.info("Loading and validating configuration...")
        packaging_cfg = load_config(args.config)
        validate_packaging_params(packaging_cfg)
        
        # Log configuration summary
        enabled_methods = get_enabled_methods(packaging_cfg)
        active_methods = [method for method, enabled in enabled_methods.items() if enabled]
        logger.info(f"Config | Run: {packaging_cfg.wandb.source_run_id} | Project: {packaging_cfg.wandb.project} | Methods: {', '.join(active_methods)} | Output: {packaging_cfg.output.output_dir}")
        
        # Run the packaging pipeline
        logger.info("Starting packaging pipeline...")
        results = run_packaging_pipeline(packaging_cfg)
        
        # Log final results
        logger.info(f"PIPELINE COMPLETED | Variants: {results.get('variants_created', 0)} | Location: {packaging_cfg.output.output_dir}")
        return 0
                
    except KeyboardInterrupt:
        logger.error("Pipeline execution was interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code != 0:
            raise RuntimeError(f"Pipeline failed with exit code {exit_code}")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise 