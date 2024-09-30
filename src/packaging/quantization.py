"""
Model Quantization Module for Transformer Models
================================================

This module implements quantization techniques to reduce model precision from FP32 to INT8,
significantly reducing model size and increasing inference speed while maintaining acceptable accuracy.

WHAT IS MODEL QUANTIZATION?
===========================
Quantization reduces the precision of model weights and activations from 32-bit floating point
to 8-bit integers. This provides several benefits:
- 4x smaller model size (32 bits → 8 bits)
- Faster inference (integer operations are faster)
- Lower memory bandwidth requirements
- Better deployment on edge devices

TYPES OF QUANTIZATION IMPLEMENTED:
=================================
1. Post-Training Quantization (PTQ): Quantize an already-trained model
2. Quantization-Aware Training (QAT): Train with quantization simulation

HOW IT WORKS:
============
1. Calibration: Analyze activation ranges on representative data
2. Scale/Zero-point computation: Map FP32 range to INT8 range
3. Quantization: Convert weights and activations to INT8
4. Dequantization: Convert back to FP32 for computation (if needed)

PYTORCH QUANTIZATION BACKEND:
============================
We use PyTorch's native quantization which supports:
- Static quantization: Requires calibration dataset
- Dynamic quantization: Quantizes weights only, activations remain FP32
- QAT: Trains with fake quantization to prepare for real quantization
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.quantization.quantize_fx as quantize_fx
import logging
import os
import wandb
from typing import Dict, Any, Optional, Tuple, List
import copy
import time

from src.models.transformer import TransformerModel
from src.training.utils import evaluate_validation_loss

logger = logging.getLogger(__name__)


class CalibrationDataset:
    """
    Dataset wrapper for calibration during post-training quantization.
    
    This class provides a subset of data to calibrate quantization parameters.
    Calibration helps determine the optimal scale and zero-point values for
    converting between FP32 and INT8 representations.
    """
    
    def __init__(self, dataloader, num_samples: int):
        """
        Initialize calibration dataset.
        
        Args:
            dataloader: Original data loader
            num_samples: Number of samples to use for calibration
        """
        self.dataloader = dataloader
        self.num_samples = num_samples
        self.samples_collected = 0
        
    def __iter__(self):
        """Iterate through calibration samples."""
        self.samples_collected = 0
        
        # Create data generator from the loader function
        data_gen = self.dataloader()
        
        while self.samples_collected < self.num_samples:
            try:
                batch = next(data_gen)
                # Return only inputs for calibration (no targets needed)
                inputs, _ = batch
                yield inputs
                
                self.samples_collected += inputs.size(0)
            except StopIteration:
                break


class ModelQuantizer:
    """
    Main class for performing model quantization on transformer models.
    
    This class handles different quantization approaches:
    1. Post-Training Quantization (PTQ): Quick quantization of trained models
    2. Quantization-Aware Training (QAT): Training with quantization simulation
    3. Dynamic Quantization: Runtime quantization of weights only
    
    The quantizer supports both static and dynamic quantization schemes
    and can handle different quantization backends for various hardware targets.
    """
    
    def __init__(self, config, device: str = "auto"):
        """
        Initialize the model quantizer.
        
        Args:
            config: Quantization configuration SimpleNamespace
            device: Computing device ("auto", "cuda", "cpu")
        """
        self.config = config
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"ModelQuantizer initialized on device: {self.device}")
        
        # Set quantization parameters
        self.backend = getattr(config.ptq, 'backend', 'fbgemm')
        self.calibration_samples = getattr(config.ptq, 'calibration_samples', 100)
        
        # For QAT, use distillation parameters if available, otherwise defaults
        if hasattr(config, 'qat'):
            self.qat_batch_size = getattr(config.qat, 'batch_size', 32)
        else:
            self.qat_batch_size = 32
            
        # QAT training parameters will be passed from distillation config when needed
        self.qat_epochs = None
        self.qat_learning_rate = None
        
    def set_qat_parameters(self, epochs: int, learning_rate: float):
        """
        Set QAT training parameters (used for combined distillation + quantization).
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for QAT
        """
        self.qat_epochs = epochs
        self.qat_learning_rate = learning_rate
        logger.info(f"QAT parameters set: epochs={epochs}, lr={learning_rate}")
        
    def prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        """
        Prepare a model for quantization by adding QuantStub and DeQuantStub.
        
        QuantStub and DeQuantStub mark the boundaries where quantization should
        happen. They convert between FP32 and INT8 representations.
        
        Args:
            model: Model to prepare for quantization
            
        Returns:
            Model wrapped with quantization stubs
        """
        # Move model to CPU for quantization
        model = model.cpu()
        
        # Create a wrapper that adds quantization boundaries
        class QuantizableWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.quant = torch.quantization.QuantStub()  # Input quantization
                self.model = original_model
                self.dequant = torch.quantization.DeQuantStub()  # Output dequantization
                
            def forward(self, x):
                x = self.quant(x)  # Convert input to quantized
                x = self.model(x)  # Run through quantized model
                x = self.dequant(x)  # Convert output back to float
                return x
                
        wrapped_model = QuantizableWrapper(model)
        
        logger.info("Model prepared for quantization with QuantStub/DeQuantStub")
        return wrapped_model
        
    def post_training_quantize(self, model: nn.Module, calibration_dataloader,
                             save_path: Optional[str] = None) -> nn.Module:
        """
        Perform post-training quantization (PTQ) on a trained model.
        
        PTQ quantizes a model without additional training. It uses a calibration
        dataset to determine optimal quantization parameters.
        
        Args:
            model: Trained model to quantize
            calibration_dataloader: Data for calibration
            save_path: Optional path to save quantized model
            
        Returns:
            Quantized model
        """
        logger.info("Starting post-training quantization...")
        
        # Prepare model for quantization
        model = self.prepare_model_for_quantization(model)
        model.eval()
        
        # Set quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig(self.backend)
        
        # Prepare for calibration
        torch.quantization.prepare(model, inplace=True)
        
        logger.info(f"Calibrating with {self.calibration_samples} samples...")
        
        # Calibration phase - run forward passes to collect statistics
        calibration_dataset = CalibrationDataset(
            calibration_dataloader, 
            self.calibration_samples
        )
        
        with torch.no_grad():
            for inputs in calibration_dataset:
                # Ensure inputs are long type for embedding layers and float32 for calibration
                inputs = inputs.to(self.device).long()
                _ = model(inputs)  # Forward pass for calibration
                
        logger.info("Calibration completed")
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        # Log quantization results
        self._log_quantization_results(model, quantized_model, "PTQ")
        
        # Save if path provided
        if save_path:
            self._save_quantized_model(quantized_model, save_path, {
                "quantization_type": "post_training",
                "backend": self.backend,
                "calibration_samples": self.calibration_samples
            })
            
        logger.info("Post-training quantization completed successfully")
        return quantized_model
        
    def quantization_aware_training(self, model: nn.Module, train_dataloader,
                                   val_dataloader, save_path: Optional[str] = None) -> nn.Module:
        """
        Perform quantization-aware training (QAT) on a model.
        
        QAT fine-tunes a model with quantization simulation, allowing it to
        adapt to quantization noise and maintain better accuracy.
        
        Args:
            model: Model to fine-tune with QAT
            train_dataloader: Training data
            val_dataloader: Validation data
            save_path: Optional path to save quantized model
            
        Returns:
            Quantized model after QAT
        """
        logger.info("Starting quantization-aware training...")
        
        # Prepare model for QAT
        model = self.prepare_model_for_quantization(model)
        model.train()
        
        # Set QAT configuration
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        
        # Prepare for QAT
        torch.quantization.prepare_qat(model, inplace=True)
        
        # Initialize optimizer for fine-tuning
        # Use default values if QAT parameters not set
        qat_lr = self.qat_learning_rate if self.qat_learning_rate is not None else 5e-5
        qat_epochs = self.qat_epochs if self.qat_epochs is not None else 3
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=qat_lr,
            weight_decay=0.01,
            eps=1e-8  # Explicitly set eps as float
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Initialize W&B for QAT tracking
        timestamp = int(time.time())
        run_name = f"qat_{timestamp}"
        wandb.init(
            project="Transformer_e2e",
            name=run_name,
            tags=["quantization", "qat", "post_training"],
            config={
                "optimization_type": "quantization_aware_training",
                "qat_epochs": qat_epochs,
                "qat_learning_rate": qat_lr,
                "backend": self.backend
            }
        )
        
        try:
            # Fine-tuning loop with quantization simulation
            for epoch in range(qat_epochs):
                logger.info(f"QAT Epoch {epoch + 1}/{qat_epochs}")
                
                model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                # Create data generator from the loader function
                train_data_gen = train_dataloader()
                
                for batch_idx in range(100):  # Process limited batches per epoch
                    try:
                        inputs, targets = next(train_data_gen)
                        # Ensure inputs are long type for embedding layers
                        inputs, targets = inputs.to(self.device).long(), targets.to(self.device).long()
                        
                        # Forward pass with quantization simulation
                        outputs = model(inputs)
                        
                        # Compute loss
                        loss = criterion(
                            outputs.view(-1, outputs.size(-1)),
                            targets.view(-1)
                        )
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        
                        optimizer.step()
                        
                        # Track metrics
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        # Log to W&B
                        wandb.log({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "qat_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })
                        
                    except StopIteration:
                        logger.info(f"Exhausted training data at batch {batch_idx}")
                        break
                        
                # Validation evaluation
                val_loss = self._evaluate_qat_model(model, val_dataloader)
                
                wandb.log({
                    "epoch": epoch,
                    "epoch_train_loss": epoch_loss / num_batches,
                    "val_loss": val_loss
                })
                
                logger.info(f"Epoch {epoch + 1} - Train Loss: {epoch_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}")
                
            # Convert to quantized model
            model.eval()
            quantized_model = torch.quantization.convert(model, inplace=False)
            
            # Log quantization results
            self._log_quantization_results(model, quantized_model, "QAT")
            
            # Save if path provided
            if save_path:
                self._save_quantized_model(quantized_model, save_path, {
                    "quantization_type": "qat",
                    "backend": self.backend,
                    "qat_epochs": qat_epochs,
                    "qat_learning_rate": qat_lr
                })
                
            logger.info("Quantization-aware training completed successfully")
            return quantized_model
            
        finally:
            wandb.finish()
            
    def dynamic_quantize(self, model: nn.Module, save_path: Optional[str] = None) -> nn.Module:
        """
        Perform dynamic quantization on a model.
        
        Dynamic quantization quantizes weights to INT8 but keeps activations in FP32.
        This provides some benefits without requiring calibration data.
        
        Args:
            model: Model to quantize
            save_path: Optional path to save quantized model
            
        Returns:
            Dynamically quantized model
        """
        logger.info("Starting dynamic quantization...")
        
        # Move model to CPU for quantization
        model = model.cpu().eval()
        
        # Define which layers to quantize
        # For transformers, we typically quantize Linear layers
        qconfig_dict = {
            nn.Linear: torch.quantization.default_dynamic_qconfig
        }
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_dict,
            dtype=torch.qint8
        )
        
        # Log quantization results
        self._log_quantization_results(model, quantized_model, "Dynamic")
        
        # Save if path provided
        if save_path:
            self._save_quantized_model(quantized_model, save_path, {
                "quantization_type": "dynamic",
                "backend": self.backend
            })
            
        logger.info("Dynamic quantization completed successfully")
        return quantized_model
        
    def _evaluate_qat_model(self, model: nn.Module, val_dataloader) -> float:
        """
        Evaluate model during QAT training.
        
        Args:
            model: Model to evaluate
            val_dataloader: Validation data loader function
            
        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        # Create validation data generator
        val_data_gen = val_dataloader()
        
        with torch.no_grad():
            # Evaluate on a limited number of validation batches
            for _ in range(20):  # Limit validation to prevent long evaluation
                try:
                    inputs, targets = next(val_data_gen)
                    # Ensure inputs are long type for embedding layers
                    inputs, targets = inputs.to(self.device).long(), targets.to(self.device).long()
                    
                    outputs = model(inputs)
                    loss = criterion(
                        outputs.view(-1, outputs.size(-1)),
                        targets.view(-1)
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                except StopIteration:
                    break
                    
        model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def _log_quantization_results(self, original_model: nn.Module, 
                                 quantized_model: nn.Module, method: str) -> None:
        """
        Log quantization results including model size and compression ratio.
        
        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            method: Quantization method name
        """
        # Calculate model sizes
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        # Calculate compression ratio
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        size_reduction = (1 - quantized_size/original_size)*100 if original_size > 0 else 0
        
        logger.info(f"{method} quantization | {original_size:.2f}MB → {quantized_size:.2f}MB | {compression_ratio:.2f}x compression | {size_reduction:.1f}% reduction")
        
    def _get_model_size(self, model: nn.Module) -> float:
        """
        Calculate model size in megabytes.
        
        Args:
            model: Model to measure
            
        Returns:
            Model size in MB
        """
        # Save model to temporary buffer to measure size
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.tell() / (1024 * 1024)  # Convert bytes to MB
        return size_mb
        
    def _save_quantized_model(self, model: nn.Module, save_path: str,
                             metadata: Dict[str, Any]) -> None:
        """
        Save quantized model with metadata.
        
        Args:
            model: Quantized model to save
            save_path: Path to save the model
            metadata: Additional metadata
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # For quantized models, we need to save both the model and its metadata
        checkpoint = {
            "model": model,  # Save the entire quantized model
            "metadata": metadata,
            "quantization_config": {
                "backend": self.backend,
                "calibration_samples": self.calibration_samples
            }
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved quantized model to {save_path}")
        
    def quantize_model_variants(self, model: nn.Module, train_dataloader,
                               val_dataloader, save_dir: str) -> Dict[str, Any]:
        """
        Create multiple quantized variants of a model.
        
        This method creates different quantized versions:
        1. Dynamic quantization (fastest to apply)
        2. Post-training quantization (requires calibration)
        3. Quantization-aware training (best quality)
        
        Args:
            model: Original model to quantize
            train_dataloader: Training data for QAT
            val_dataloader: Validation data for calibration and evaluation
            save_dir: Directory to save quantized variants
            
        Returns:
            Dictionary containing results for each quantization method
        """
        logger.info("Creating quantized variants | Dynamic, PTQ, QAT...")
        
        results = {}
        
        # 1. Dynamic Quantization
        logger.info("Running dynamic quantization...")
        try:
            dynamic_model = self.dynamic_quantize(
                copy.deepcopy(model),
                os.path.join(save_dir, "dynamic_quantized.pt")
            )
            compression_ratio = self._get_compression_ratio(model, dynamic_model)
            results["dynamic"] = {
                "success": True,
                "model": dynamic_model,
                "compression_ratio": compression_ratio
            }
            logger.info(f"✓ Dynamic quantization | Compression: {compression_ratio:.2f}x")
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {str(e)}")
            results["dynamic"] = {"success": False, "error": str(e)}
            
        # 2. Post-Training Quantization
        logger.info("Running post-training quantization...")
        try:
            ptq_model = self.post_training_quantize(
                copy.deepcopy(model),
                val_dataloader,
                os.path.join(save_dir, "ptq_quantized.pt")
            )
            compression_ratio = self._get_compression_ratio(model, ptq_model)
            results["ptq"] = {
                "success": True,
                "model": ptq_model,
                "compression_ratio": compression_ratio
            }
            logger.info(f"✓ Post-training quantization | Compression: {compression_ratio:.2f}x")
        except Exception as e:
            logger.error(f"Post-training quantization failed: {str(e)}")
            results["ptq"] = {"success": False, "error": str(e)}
            
        # 3. Quantization-Aware Training
        logger.info("Running quantization-aware training...")
        try:
            qat_model = self.quantization_aware_training(
                copy.deepcopy(model),
                train_dataloader,
                val_dataloader,
                os.path.join(save_dir, "qat_quantized.pt")
            )
            compression_ratio = self._get_compression_ratio(model, qat_model)
            results["qat"] = {
                "success": True,
                "model": qat_model,
                "compression_ratio": compression_ratio
            }
            logger.info(f"✓ Quantization-aware training | Compression: {compression_ratio:.2f}x")
        except Exception as e:
            logger.error(f"Quantization-aware training failed: {str(e)}")
            results["qat"] = {"success": False, "error": str(e)}
            
        successful_methods = [k for k, v in results.items() if v.get("success", False)]
        logger.info(f"Quantization completed | Methods: {len(successful_methods)}/{len(results)} successful")
        return results
        
    def _get_compression_ratio(self, original_model: nn.Module, 
                              quantized_model: nn.Module) -> float:
        """
        Calculate compression ratio between original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Compression ratio (original_size / quantized_size)
        """
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        return original_size / quantized_size if quantized_size > 0 else 1.0 