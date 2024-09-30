"""
Quantization-Aware Distillation Module for Transformer Models
============================================================

This module implements quantization-aware distillation (QAD), a hybrid optimization
technique that combines knowledge distillation with quantization simulation.

WHAT IS QUANTIZATION-AWARE DISTILLATION?
=======================================
QAD is a hybrid technique that trains smaller student models to mimic larger teacher
models while simultaneously learning to handle quantization noise. This results in
models that are both smaller (from distillation) and quantized (INT8) with minimal
accuracy loss.

HOW IT WORKS:
============
1. Teacher Model: Large, well-trained transformer (FP32)
2. Student Model: Smaller architecture with quantization simulation enabled
3. Training: Student learns from teacher while adapting to quantization noise
4. Output: Small, quantized model with optimal accuracy retention

BENEFITS OVER SEQUENTIAL APPROACH:
=================================
- Better accuracy than distillation → quantization
- Student learns quantization-robust features from the start
- Single training process instead of two separate steps
- Optimal trade-off between size, speed, and accuracy

WHEN TO USE QAD:
===============
- Need maximum compression (small + quantized)
- Deploy to resource-constrained environments
- Want best accuracy for combined optimization
- Have training budget for hybrid approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import time
from typing import Dict, Tuple, Optional, Any, List

from src.models.transformer import TransformerModel

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Custom loss function for knowledge distillation.
    
    This combines two types of losses:
    1. Distillation Loss: How well student matches teacher's soft predictions
    2. Ground Truth Loss: How well student predicts actual labels
    """
    
    def __init__(self, temperature: float, alpha: float):
        """
        Initialize the distillation loss function.
        
        Args:
            temperature: Controls softness of probability distributions
            alpha: Weight for balancing distillation vs ground truth loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined distillation loss.
        
        Args:
            student_logits: Raw predictions from student model [batch, seq_len, vocab]
            teacher_logits: Raw predictions from teacher model [batch, seq_len, vocab]
            targets: Ground truth token indices [batch, seq_len]
            
        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        # Reshape for loss computation: [batch * seq_len, vocab_size]
        batch_size, seq_len, vocab_size = student_logits.shape
        
        student_flat = student_logits.view(-1, vocab_size)
        teacher_flat = teacher_logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Ground truth loss
        ground_truth_loss = self.criterion(student_flat, targets_flat)
        
        # Distillation loss with temperature scaling
        student_soft = F.log_softmax(student_flat / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            student_soft, teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = (
            self.alpha * distillation_loss + 
            (1.0 - self.alpha) * ground_truth_loss
        )
        
        loss_components = {
            "total_loss": total_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "ground_truth_loss": ground_truth_loss.item(),
            "distillation_weight": self.alpha,
            "ground_truth_weight": 1.0 - self.alpha
        }
        
        return total_loss, loss_components


class QuantizationAwareDistiller:
    """
    Main class for performing quantization-aware distillation on transformer models.
    
    This class handles the complete QAD pipeline:
    1. Preparing student models for quantization simulation
    2. Training students with both teacher knowledge and quantization awareness
    3. Converting QAT models to fully quantized models
    4. Saving and evaluating QAD models
    """
    
    def __init__(self, combined_config, device: str = "auto"):
        """
        Initialize the quantization-aware distiller.
        
        Args:
            combined_config: Combined optimization configuration SimpleNamespace
            device: Computing device ("auto", "cuda", "cpu")
        """
        self.config = combined_config
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"QuantizationAwareDistiller initialized on device: {self.device}")
        
        # Initialize models (will be set later)
        self.teacher_model = None
        self.student_models = {}
        
        # QAD-specific distillation loss with optimized parameters
        self.qad_loss = DistillationLoss(
            temperature=combined_config.temperature,  # Higher temperature for QAD
            alpha=combined_config.alpha  # Higher alpha for more teacher influence
        )
        
    def set_teacher_model(self, teacher_model: nn.Module) -> None:
        """
        Set the teacher model for QAD.
        
        Args:
            teacher_model: Pre-loaded teacher model
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Ensure it's in evaluation mode
        logger.info(f"Teacher model set for QAD | Parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        
    def create_student_models_from_architectures(self, architectures: Dict[str, Dict[str, Any]], 
                                               vocab_size: int) -> None:
        """
        Create student models from pre-calculated architectures for QAD.
        
        Args:
            architectures: Dictionary of student architectures with calculated dimensions
            vocab_size: Vocabulary size for the models
        """
        logger.info("Creating student models for quantization-aware distillation...")
        
        for student_name, arch_config in architectures.items():
            logger.info(f"Creating QAD student model: {student_name} with {arch_config}")
            
            # Create student model with calculated architecture
            student_model = TransformerModel(
                vocab_size=vocab_size,
                context_window=arch_config["context_window"],
                channel_dim=arch_config["channel_dim"],
                num_heads=arch_config["num_heads"],
                num_layers=arch_config["num_layers"],
                dropout_rate=arch_config["dropout_rate"],
                final_dropout_multiplier=getattr(self.config, 'final_dropout_multiplier', None),
                max_dropout_val=getattr(self.config, 'max_dropout_val', 0.5)
            ).to(self.device)
            
            self.student_models[student_name] = student_model
            
            # Log model size for comparison
            num_params = sum(p.numel() for p in student_model.parameters())
            logger.info(f"QAD {student_name.capitalize()} student parameters: {num_params:,}")
            
        # Log compression ratios for QAD
        if self.teacher_model is not None:
            teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            logger.info(f"Teacher model parameters: {teacher_params:,}")
            
            for name, model in self.student_models.items():
                student_params = sum(p.numel() for p in model.parameters())
                compression_ratio = teacher_params / student_params
                logger.info(f"QAD {name.capitalize()} compression ratio: {compression_ratio:.2f}x")
                
    def distill_student_with_quantization(self, student_name: str, train_dataloader, 
                                         val_dataloader, save_dir: str, 
                                         quantization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform quantization-aware distillation for a specific student model.
        
        Args:
            student_name: Name of the student architecture
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            save_dir: Directory to save the distilled model
            quantization_config: Quantization configuration
            
        Returns:
            Dictionary containing training metrics and results
        """
        logger.info(f"Starting quantization-aware distillation for {student_name} student model...")
        
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded. Call set_teacher_model first.")
            
        if student_name not in self.student_models:
            raise ValueError(f"Student model '{student_name}' not found.")
            
        student_model = self.student_models[student_name]
        
        # Prepare student model for quantization-aware training
        student_model = self._prepare_model_for_qat(student_model, quantization_config)
        
        # Initialize QAD-specific optimizer and scheduler
        optimizer = self._create_qad_optimizer(student_model)
        scheduler = self._create_qad_scheduler(optimizer)
        
        # Training metrics tracking
        best_val_loss = float('inf')
        patience_counter = 0
        step = 0
        training_metrics = []
        
        # QAD training loop with more steps due to quantization complexity
        max_steps = int(self.config.num_epochs * 100)  # More steps for QAD convergence
        
        # Create data generator from the loader function
        train_data_gen = train_dataloader()
        
        for step in range(max_steps):
            student_model.train()
            
            # Get a batch from the generator
            try:
                batch = next(train_data_gen)
                if isinstance(batch, tuple) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    logger.error(f"Unexpected batch format: {type(batch)}")
                    break
            except StopIteration:
                logger.info(f"Exhausted training data at step {step}")
                break
                
            # Ensure inputs are long type for embedding layers
            inputs, targets = inputs.to(self.device).long(), targets.to(self.device).long()
            
            # Get predictions from both models
            with torch.no_grad():
                teacher_logits, _ = self.teacher_model(inputs)
                
            student_logits, _ = student_model(inputs)
            
            # Compute QAD loss
            loss, loss_components = self.qad_loss(
                student_logits, teacher_logits, targets
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability (important for QAD)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            
            optimizer.step()
            
            # Store metrics for manager to log
            step_metrics = {
                "step": step,
                "train_loss": loss.item(),
                "distillation_loss": loss_components["distillation_loss"],
                "ground_truth_loss": loss_components["ground_truth_loss"],
                "learning_rate": optimizer.param_groups[0]['lr'],
                "qad_enabled": True,
                "quantization_backend": quantization_config.get('backend', 'fbgemm'),
                "temperature": self.config.temperature,
                "alpha": self.config.alpha
            }
            training_metrics.append(step_metrics)
            
            # Validation check every 100 steps
            if step % 100 == 0:
                val_loss = self._evaluate_qad_student(student_model, val_dataloader)
                
                step_metrics["val_loss"] = val_loss
                
                logger.info(f"QAD Step {step}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping and model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model (convert to quantized first)
                    quantized_model = self._convert_qat_to_quantized(student_model)
                    save_path = os.path.join(save_dir, f"{student_name}_qad_best.pt")
                    self._save_qad_model(quantized_model, save_path, {
                        "step": step,
                        "val_loss": val_loss,
                        "student_name": student_name,
                        "quantization_aware": True,
                        "backend": quantization_config.get('backend', 'fbgemm'),
                        "qad_config": {
                            "temperature": self.config.temperature,
                            "alpha": self.config.alpha,
                            "learning_rate": self.config.learning_rate
                        }
                    })
                    
                    logger.info(f"New best QAD validation loss: {val_loss:.4f} at step {step}")
                else:
                    patience_counter += 1
                    
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping check
                if patience_counter >= self.config.patience:
                    logger.info(f"QAD early stopping triggered after {step} steps")
                    break
                
        # Convert final model to quantized
        final_quantized_model = self._convert_qat_to_quantized(student_model)
        
        # Training completed
        training_results = {
            "final_step": step,
            "best_val_loss": best_val_loss,
            "final_epoch": step // 100,  # Approximate epochs based on validation frequency
            "student_name": student_name,
            "quantization_aware": True,
            "optimization_method": "quantization_aware_distillation",
            "compression_ratio": sum(p.numel() for p in self.teacher_model.parameters()) / 
                               sum(p.numel() for p in final_quantized_model.parameters()),
            "model": final_quantized_model,  # Return the quantized model
            "training_metrics": training_metrics,  # For manager to log to W&B
            "qad_config": {
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "learning_rate": self.config.learning_rate,
                "backend": quantization_config.get('backend', 'fbgemm')
            }
        }
        
        logger.info(f"Quantization-aware distillation completed for {student_name}: Best Val Loss = {best_val_loss:.4f}")
        
        return training_results
    
    def _create_qad_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create QAD-specific optimizer with conservative parameters."""
        optimizer_config = getattr(self.config, 'optimizer', None)
        
        if optimizer_config and hasattr(optimizer_config, 'type'):
            if optimizer_config.type == "AdamW":
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=getattr(optimizer_config, 'weight_decay', 0.005),
                    betas=getattr(optimizer_config, 'betas', [0.9, 0.999]),
                    eps=getattr(optimizer_config, 'eps', 1e-8)
                )
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=0.005
                )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.005
            )
            
        logger.info(f"Created QAD optimizer: lr={self.config.learning_rate}")
        return optimizer
    
    def _create_qad_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Create QAD-specific learning rate scheduler."""
        scheduler_config = getattr(self.config, 'scheduler', None)
        
        if scheduler_config and hasattr(scheduler_config, 'type'):
            if scheduler_config.type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=getattr(scheduler_config, 'mode', 'min'),
                    factor=getattr(scheduler_config, 'factor', 0.7),
                    patience=getattr(scheduler_config, 'patience', 7),
                    verbose=getattr(scheduler_config, 'verbose', True)
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.7, patience=7, verbose=True
                )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=7, verbose=True
            )
            
        return scheduler
    
    def _prepare_model_for_qat(self, model: nn.Module, quantization_config: Dict[str, Any]) -> nn.Module:
        """Prepare a model for quantization-aware training."""
        backend = quantization_config.get('backend', 'fbgemm')
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(model, inplace=True)
        logger.info(f"Model prepared for QAT with backend: {backend}")
        return model
    
    def _convert_qat_to_quantized(self, qat_model: nn.Module) -> nn.Module:
        """Convert a QAT model to a fully quantized model."""
        qat_model.eval()
        quantized_model = torch.quantization.convert(qat_model, inplace=False)
        logger.info("Converted QAT model to fully quantized model")
        return quantized_model
    
    def _evaluate_qad_student(self, student_model: nn.Module, val_dataloader) -> float:
        """Evaluate QAD student model on validation set."""
        student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Create validation data generator
        val_data_gen = val_dataloader()
        
        with torch.no_grad():
            # Evaluate on a limited number of validation batches
            for _ in range(10):  # Evaluate on 10 batches for efficiency
                try:
                    batch = next(val_data_gen)
                    if isinstance(batch, tuple) and len(batch) == 2:
                        inputs, targets = batch
                    else:
                        logger.error(f"Unexpected validation batch format: {type(batch)}")
                        break
                    # Ensure inputs are long type for embedding layers
                    inputs, targets = inputs.to(self.device).long(), targets.to(self.device).long()
                    student_logits, _ = student_model(inputs)
                    loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        targets.view(-1)
                    )
                    total_loss += loss.item()
                    num_batches += 1
                except StopIteration:
                    break
                
        student_model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_qad_model(self, model: nn.Module, save_path: str, metadata: Dict[str, Any]) -> None:
        """Save QAD model checkpoint with metadata."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "qad_config": {
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "learning_rate": self.config.learning_rate
            },
            "optimization_method": "quantization_aware_distillation"
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved QAD model checkpoint to {save_path}")
    
    def distill_all_students_with_quantization(self, train_dataloader, val_dataloader, 
                                             save_dir: str, quantization_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Perform quantization-aware distillation for all student architectures."""
        logger.info("Starting quantization-aware distillation for all students...")
        
        if not self.student_models:
            raise ValueError("No student models created. Call create_student_models_from_architectures first.")
            
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        for student_name in self.student_models.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"QAD TRAINING: {student_name.upper()} STUDENT")
            logger.info(f"{'='*60}")
            
            try:
                student_results = self.distill_student_with_quantization(
                    student_name, train_dataloader, val_dataloader, save_dir, quantization_config
                )
                results[student_name] = student_results
                logger.info(f"✓ QAD completed for {student_name}: Val Loss = {student_results['best_val_loss']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed QAD for {student_name} student: {str(e)}")
                results[student_name] = {"error": str(e)}
                
        logger.info("\nQuantization-aware distillation completed for all students!")
        return results 