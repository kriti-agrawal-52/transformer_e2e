"""
Model Packaging Module for Transformer Models
=============================================

This module provides comprehensive model optimization and packaging capabilities
for transformer models, including knowledge distillation, quantization, 
benchmarking, and deployment preparation.

MAIN COMPONENTS:
===============
- ModelDistiller: Knowledge distillation for creating smaller models
- ModelQuantizer: INT8 quantization for faster inference  
- ModelBenchmarker: Comprehensive performance evaluation
- ModelVariantsManager: Orchestrates the complete packaging pipeline

TYPICAL USAGE:
=============
```python
from src.packaging import ModelVariantsManager, PackagingConfig

# Configure packaging pipeline
config = PackagingConfig(
    wandb_project="your_project",
    run_id="your_run_id",
    output_dir="packaged_models"
)

# Create and run packaging pipeline
manager = ModelVariantsManager(config)
results = manager.run_complete_packaging_pipeline(
    model_config, train_dataloader, val_dataloader, eval_dataloader, tokenizer
)
```

OPTIMIZATION TECHNIQUES:
=======================
1. Knowledge Distillation: Teacher-student training for smaller models
2. Model Quantization: INT8 precision reduction for faster inference
3. Combined Optimization: Both distillation and quantization
4. Comprehensive Benchmarking: Quality and performance evaluation

OUTPUTS:
=======
- Optimized model variants (4 types: baseline, distilled, quantized, combined)
- Performance benchmarks and comparisons
- Deployment recommendations for different scenarios
- Complete packaging reports and documentation
"""

from .distillation import ModelDistiller, DistillationLoss
from .quantization import ModelQuantizer, CalibrationDataset
from .benchmarking import ModelBenchmarker, BenchmarkResults
from .model_variants import ModelVariantsManager, ModelVariant
from .utils import (
    validate_packaging_config,
    load_model_from_wandb,
    save_model_with_metadata,
    create_packaging_workspace,
    export_model_for_inference,
    create_deployment_package,
    run_model_smoke_test
)

# Version information
__version__ = "0.1.0"
__author__ = "Kriti Agrawal"

# Main exports for easy importing
__all__ = [
    # Main orchestrator
    "ModelVariantsManager",
    
    # Core optimization classes
    "ModelDistiller",
    "ModelQuantizer", 
    "ModelBenchmarker",
    
    # Data structures
    "ModelVariant",
    "BenchmarkResults",
    "DistillationLoss",
    "CalibrationDataset",
    
    # Utility functions
    "validate_packaging_config",
    "load_model_from_wandb",
    "save_model_with_metadata",
    "create_packaging_workspace",
    "export_model_for_inference", 
    "create_deployment_package",
    "run_model_smoke_test"
]

# Module metadata
OPTIMIZATION_METHODS = ["distillation", "quantization", "combined"]
SUPPORTED_FORMATS = ["torchscript", "onnx", "pt"]
VARIANT_TYPES = ["baseline", "distilled", "quantized", "combined"]
