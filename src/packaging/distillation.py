# src/packaging/distillation.py
"""
Knowledge Distillation Module for Transformer Models
====================================================

This module implements knowledge distillation techniques to compress large transformer models
into smaller, more efficient variants while preserving as much performance as possible.

WHAT IS KNOWLEDGE DISTILLATION?
==============================
Knowledge distillation is a technique where a large, well-trained model (teacher) transfers
its knowledge to a smaller model (student). The student learns not just from the ground truth
labels, but also from the teacher's "soft" predictions, which contain richer information
about the relationships between different outputs.

HOW IT WORKS:
============
1. Teacher Model: Large, well-trained transformer (e.g., 8 layers, 8 heads)
2. Student Model: Smaller architecture (e.g., 4 layers, 4 heads)
3. Training: Student learns from both ground truth and teacher's predictions
4. Loss: Combination of standard cross-entropy and distillation loss

BENEFITS:
========
- Smaller model size (fewer parameters)
- Faster inference (less computation)
- Maintained performance (retains teacher's knowledge)
- Better generalization (soft targets provide more information)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import wandb
import os
import time
import shutil
import json
import threading
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path

from src.models.transformer import TransformerModel
from src.training.utils import evaluate_validation_loss
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Memory optimization constants
MEMORY_EFFICIENT_BATCH_SIZE = 8  # Smaller batch size for memory-constrained environments
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients to simulate larger batch size
MAX_MEMORY_USAGE_GB = 12.0  # Maximum GPU memory usage threshold

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0

def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def should_use_memory_efficient_mode():
    """Determine if we should use memory-efficient training mode."""
    if not torch.cuda.is_available():
        return False
    
    # Check total GPU memory
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    current_memory_gb = get_gpu_memory_usage()
    logger.info(f"GPU total memory: {total_memory_gb:.2f} GB, current usage: {current_memory_gb:.2f} GB")
    
    # Use memory-efficient mode for GPUs with less than 16GB
    return total_memory_gb < 16.0

def log_memory_usage(step: int, context: str = ""):
    """Log current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Step {step} {context} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

# W&B upload functionality moved to ModelVariantsManager


def cleanup_training_artifacts(save_dir: str, keep_best: bool = True) -> None:
    """
    Clean up training artifacts and intermediate checkpoints.
    
    Args:
        save_dir: Directory containing training artifacts
        keep_best: Whether to keep the best model checkpoint
    """
    if not os.path.exists(save_dir):
        return
        
    logger.info(f"Cleaning up training artifacts in {save_dir}")
    
    # List of patterns to clean up
    cleanup_patterns = [
        "*.tmp",
        "*.temp", 
        "*_intermediate_*.pt",
        "*_epoch_*.pt",
        "optimizer_*.pt",
        "scheduler_*.pt"
    ]
    
    files_removed = 0
    for pattern in cleanup_patterns:
        import glob
        for file_path in glob.glob(os.path.join(save_dir, pattern)):
            try:
                os.remove(file_path)
                files_removed += 1
                logger.debug(f"Removed: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")
    
    # Remove empty subdirectories
    for root, dirs, files in os.walk(save_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Empty directory
                    os.rmdir(dir_path)
                    logger.debug(f"Removed empty directory: {dir_path}")
            except Exception as e:
                logger.debug(f"Could not remove directory {dir_path}: {str(e)}")
    
    if files_removed > 0:
        logger.info(f"Cleaned up {files_removed} training artifacts")
    else:
        logger.debug("No training artifacts found to clean up")


class DistillationLoss(nn.Module):
    """
    Custom loss function for knowledge distillation.
    
    This combines two types of losses:
    1. Distillation Loss: How well student matches teacher's soft predictions
    2. Ground Truth Loss: How well student predicts actual labels
    
    The combination helps the student learn both from the teacher's knowledge
    and from the true data distribution.
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
        
        # Standard cross-entropy for ground truth comparison
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined distillation loss with memory optimization.
        
        Args:
            student_logits: Raw predictions from student model [batch, seq_len, vocab]
            teacher_logits: Raw predictions from teacher model [batch, seq_len, vocab]
            targets: Ground truth token indices [batch, seq_len]
            
        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        # Reshape for loss computation: [batch * seq_len, vocab_size]
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Memory optimization: process in chunks if tensors are too large
        total_elements = batch_size * seq_len * vocab_size
        
        # If tensor is very large, process in chunks to avoid OOM
        # Reduced threshold for T4 GPU memory constraints
        if total_elements > 25_000_000:  # ~100MB for float32
            return self._forward_chunked(student_logits, teacher_logits, targets)
        
        student_flat = student_logits.view(-1, vocab_size)
        teacher_flat = teacher_logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # 1. GROUND TRUTH LOSS
        # Standard cross-entropy between student predictions and true labels
        ground_truth_loss = self.criterion(student_flat, targets_flat)
        
        # 2. DISTILLATION LOSS  
        # KL divergence between temperature-scaled student and teacher predictions
        
        # Apply temperature scaling to make distributions "softer"
        # Higher temperature → more uniform distribution → easier to learn patterns
        student_soft = F.log_softmax(student_flat / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=1)
        
        # KL divergence measures how different two probability distributions are
        # We want to minimize this to make student predictions similar to teacher
        distillation_loss = F.kl_div(
            student_soft, teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale by T^2 as per distillation literature
        
        # Clear intermediate tensors to save memory
        del student_soft, teacher_soft, student_flat, teacher_flat, targets_flat
        
        # 3. COMBINE LOSSES
        # Alpha controls the balance: more alpha = more emphasis on teacher knowledge
        total_loss = (
            self.alpha * distillation_loss + 
            (1.0 - self.alpha) * ground_truth_loss
        )
        
        # Return both total loss and components for monitoring
        loss_components = {
            "total_loss": total_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "ground_truth_loss": ground_truth_loss.item(),
            "distillation_weight": self.alpha,
            "ground_truth_weight": 1.0 - self.alpha
        }
        
        return total_loss, loss_components
    
    def _forward_chunked(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                        targets: torch.Tensor, chunk_size: int = 1024) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process loss computation in chunks to handle large tensors.
        
        Args:
            student_logits: Student predictions
            teacher_logits: Teacher predictions  
            targets: Ground truth targets
            chunk_size: Size of chunks to process (in sequence positions)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        total_samples = batch_size * seq_len
        
        total_ground_truth_loss = 0.0
        total_distillation_loss = 0.0
        
        # Flatten tensors once
        student_flat = student_logits.view(-1, vocab_size)
        teacher_flat = teacher_logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Process in chunks
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Extract chunk
            chunk_student = student_flat[start_idx:end_idx]
            chunk_teacher = teacher_flat[start_idx:end_idx]
            chunk_targets = targets_flat[start_idx:end_idx]
            
            # Compute losses for this chunk
            ground_truth_loss = self.criterion(chunk_student, chunk_targets)
            
            student_soft = F.log_softmax(chunk_student / self.temperature, dim=1)
            teacher_soft = F.softmax(chunk_teacher / self.temperature, dim=1)
            
            distillation_loss = F.kl_div(
                student_soft, teacher_soft, 
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Accumulate losses (weighted by chunk size)
            chunk_weight = (end_idx - start_idx) / total_samples
            total_ground_truth_loss += ground_truth_loss.item() * chunk_weight
            total_distillation_loss += distillation_loss.item() * chunk_weight
            
            # Clean up chunk tensors
            del chunk_student, chunk_teacher, chunk_targets, student_soft, teacher_soft
            del ground_truth_loss, distillation_loss
        
        # Clean up flattened tensors
        del student_flat, teacher_flat, targets_flat
        
        # Create final loss tensors
        ground_truth_loss = torch.tensor(total_ground_truth_loss, device=student_logits.device, requires_grad=True)
        distillation_loss = torch.tensor(total_distillation_loss, device=student_logits.device, requires_grad=True)
        
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


class ModelDistiller:
    """
    Main class for performing knowledge distillation on transformer models.
    
    This class handles the complete distillation pipeline:
    1. Loading teacher model from W&B
    2. Creating student models with different architectures
    3. Training students to mimic teacher behavior
    4. Saving and evaluating distilled models
    
    The distiller supports multiple student architectures simultaneously,
    allowing comparison of different size-performance trade-offs.
    """
    
    def __init__(self, config, device: str = "auto"):
        """
        Initialize the model distiller.
        
        Args:
            config: Distillation configuration SimpleNamespace
            device: Computing device ("auto", "cuda", "cpu")
        """
        self.config = config
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"ModelDistiller initialized on device: {self.device}")
        
        # Initialize models (will be loaded later)
        self.teacher_model = None
        self.student_models = {}
        
        # Training components
        self.distillation_loss = DistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha
        )
        
    def _calculate_incremental_dropout_rates(self, num_layers: int) -> List[float]:
        """
        Calculate incremental dropout rates for each layer, matching the teacher model pattern.
        
        This method implements the same progressive dropout pattern used in the original
        training, where dropout rates increase from bottom to top layers.
        
        Args:
            num_layers: Number of layers in the student model
            
        Returns:
            List of dropout rates for each layer
        """
        if num_layers == 1:
            return [self.config.dropout_rate]
        
        dropout_rates = []
        
        for layer_idx in range(num_layers):
            # Calculate progression factor (0.0 for first layer, 1.0 for last layer)
            progression = layer_idx / (num_layers - 1)
            
            # Calculate dropout rate with linear progression
            dropout_rate = self.config.dropout_rate * (
                1.0 + progression * (self.config.final_dropout_multiplier - 1.0)
            )
            
            # Apply maximum dropout cap
            dropout_rate = min(dropout_rate, self.config.max_dropout_val)
            
            dropout_rates.append(dropout_rate)
            
        logger.debug(f"Calculated dropout rates for {num_layers} layers: {dropout_rates}")
        return dropout_rates
        
    def load_teacher_from_wandb(self, run_id: str, project_name: str, 
                               model_config: Dict[str, Any]) -> None:
        """
        Load the teacher model from W&B artifacts.
        
        NOTE: This method is deprecated in favor of centralized model loading 
        in ModelVariantsManager. Use set_teacher_model() instead.
        
        Args:
            run_id: W&B run identifier containing the model
            project_name: W&B project name
            model_config: Configuration parameters for model architecture
        """
        logger.warning("load_teacher_from_wandb is deprecated. Use ModelVariantsManager for centralized model loading.")
        
        logger.info(f"Loading teacher model from W&B run: {run_id}")
        
        try:
            # Initialize W&B API
            api = wandb.Api()
            
            # Get the specific run
            run = api.run(f"{project_name}/{run_id}")
            
            # Look for model artifacts in this run
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
            
            # Find the model checkpoint file
            checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
            if not checkpoint_files:
                raise ValueError(f"No .pt checkpoint files found in artifact")
                
            checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
            
            # Create teacher model with the same architecture
            self.teacher_model = TransformerModel(
                vocab_size=model_config["vocab_size"],
                channel_dim=model_config["channel_dim"],
                context_window=model_config["context_window"],
                num_heads=model_config["num_heads"],
                num_layers=model_config["num_layers"],
                dropout_rate=model_config.get("dropout_rate", 0.2),
                final_dropout_multiplier=model_config.get("final_dropout_multiplier"),
                max_dropout_val=model_config.get("max_dropout_val", 0.5)
            ).to(self.device)
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.teacher_model.load_state_dict(checkpoint["model_state_dict"])
            
            # Set to evaluation mode (no training, no dropout)
            self.teacher_model.eval()
            
            logger.info(f"Successfully loaded teacher model from {checkpoint_path}")
            logger.info(f"Teacher model parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load teacher model: {str(e)}")
            raise
            
    def set_teacher_model(self, teacher_model: nn.Module) -> None:
        """
        Set the teacher model directly (preferred method).
        
        Args:
            teacher_model: Pre-loaded teacher model
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Ensure it's in evaluation mode
        logger.info(f"Teacher model set | Parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
            
    def create_student_models(self, base_config: Dict[str, Any]) -> None:
        """
        Create student models with different architectures.
        
        This method creates multiple student models with varying sizes,
        allowing exploration of the size-performance trade-off.
        
        The student models use the same incremental dropout pattern as the teacher
        model to maintain architectural consistency during distillation.
        
        Args:
            base_config: Base configuration (vocab_size, context_window, etc.)
        """
        logger.info("Creating student models with different architectures...")
        
        for student_name, arch_config in self.config.student_architectures.items():
            logger.info(f"Creating {student_name} student model: {arch_config}")
            
            # Calculate incremental dropout rates for this student architecture
            num_layers = arch_config["num_layers"]
            dropout_rates = self._calculate_incremental_dropout_rates(num_layers)
            
            logger.info(f"Using incremental dropout rates for {student_name}: {dropout_rates}")
            
            # Create student model with reduced architecture and incremental dropout
            student_model = TransformerModel(
                vocab_size=base_config["vocab_size"],
                context_window=base_config["context_window"],
                channel_dim=arch_config["channel_dim"],
                num_heads=arch_config["num_heads"],
                num_layers=arch_config["num_layers"],
                dropout_rate=self.config.dropout_rate,  # Base dropout rate
                final_dropout_multiplier=self.config.final_dropout_multiplier,
                max_dropout_val=self.config.max_dropout_val
            ).to(self.device)
            
            self.student_models[student_name] = student_model
            
            # Log model size for comparison
            num_params = sum(p.numel() for p in student_model.parameters())
            logger.info(f"{student_name.capitalize()} student parameters: {num_params:,}")
            
        # Log compression ratios
        if self.teacher_model is not None:
            teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            logger.info(f"Teacher model parameters: {teacher_params:,}")
            
            for name, model in self.student_models.items():
                student_params = sum(p.numel() for p in model.parameters())
                compression_ratio = teacher_params / student_params
                logger.info(f"{name.capitalize()} compression ratio: {compression_ratio:.2f}x")
    
    def create_student_models_from_architectures(self, architectures: Dict[str, Dict[str, Any]], 
                                               vocab_size: int) -> None:
        """
        Create student models from pre-calculated architectures.
        
        Args:
            architectures: Dictionary of student architectures with calculated dimensions
            vocab_size: Vocabulary size for the models
        """
        logger.info("Creating student models from calculated architectures...")
        
        for student_name, arch_config in architectures.items():
            logger.info(f"Creating {student_name} student model: {arch_config}")
            
            # Create student model with calculated architecture
            student_model = TransformerModel(
                vocab_size=vocab_size,
                context_window=arch_config["context_window"],
                channel_dim=arch_config["channel_dim"],
                num_heads=arch_config["num_heads"],
                num_layers=arch_config["num_layers"],
                dropout_rate=arch_config["dropout_rate"],
                final_dropout_multiplier=self.config.final_dropout_multiplier,
                max_dropout_val=self.config.max_dropout_val
            ).to(self.device)
            
            self.student_models[student_name] = student_model
            
            # Log model size for comparison
            num_params = sum(p.numel() for p in student_model.parameters())
            logger.info(f"{student_name.capitalize()} student parameters: {num_params:,}")
            
        # Log compression ratios
        if self.teacher_model is not None:
            teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            logger.info(f"Teacher model parameters: {teacher_params:,}")
            
            for name, model in self.student_models.items():
                student_params = sum(p.numel() for p in model.parameters())
                compression_ratio = teacher_params / student_params
                logger.info(f"{name.capitalize()} compression ratio: {compression_ratio:.2f}x")
                
    def distill_student(self, student_name: str, train_dataloader, 
                       val_dataloader, save_dir: str) -> Dict[str, Any]:
        """
        Perform knowledge distillation for a specific student model with memory optimization.
        
        This method trains a student model to mimic the teacher's behavior
        using the distillation loss function with gradient accumulation for memory efficiency.
        
        Args:
            student_name: Name of the student architecture
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            save_dir: Directory to save the distilled model
            
        Returns:
            Dictionary containing training metrics and results
        """
        logger.info(f"Starting distillation for {student_name} student model...")
        
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded. Call load_teacher_from_wandb first.")
            
        if student_name not in self.student_models:
            raise ValueError(f"Student model '{student_name}' not found.")
            
        student_model = self.student_models[student_name]
        
        # Use the same batch-wise approach as base training (no gradient accumulation needed)
        logger.info("Using batch-wise training approach (same as base model training)")
        use_memory_efficient = should_use_memory_efficient_mode()
        if use_memory_efficient:
            logger.info("Memory optimizations enabled: CPU offloading, chunked loss computation")
        
        try:
            logger.info("DEBUG: Creating optimizer...")
            # Initialize optimizer and scheduler for student
            optimizer = self._create_optimizer(student_model, self.config)
            logger.info("DEBUG: Optimizer created successfully")
            
            logger.info("DEBUG: Creating scheduler...")
            scheduler = self._create_scheduler(optimizer, self.config)
            logger.info("DEBUG: Scheduler created successfully")
            
            # Training metrics tracking
            best_val_loss = float('inf')
            patience_counter = 0
            step = 0
            training_metrics = []
            
            # Step-based training loop (not epoch-based)
            max_steps = 1000  # Reasonable limit for distillation
            
            logger.info("DEBUG: Creating data generator...")
            # Create data generator from the loader function
            train_data_gen = train_dataloader()
            logger.info("DEBUG: Data generator created successfully")
            
            logger.info("DEBUG: Starting training loop...")
            
            for step in range(max_steps):
                student_model.train()
                
                # Get a batch (same as base training approach)
                try:
                    logger.info(f"DEBUG: Getting batch for step {step}...")
                    inputs, targets = next(train_data_gen)
                    logger.info(f"DEBUG: Batch retrieved successfully for step {step}")
                except StopIteration:
                    # If we've exhausted the dataloader, break
                    logger.info(f"Exhausted training data at step {step}")
                    break
                    
                # Ensure inputs are long type for embedding layers
                inputs, targets = inputs.to(self.device).long(), targets.to(self.device).long()
                
                logger.info(f"DEBUG: Getting teacher predictions for step {step}...")
                # Get predictions from both models with memory optimization
                with torch.no_grad():
                    teacher_logits, _ = self.teacher_model(inputs, targets)
                    # Move teacher logits to CPU to free GPU memory if needed
                    if use_memory_efficient:
                        teacher_logits = teacher_logits.cpu()
                logger.info(f"DEBUG: Teacher predictions obtained for step {step}, shape: {teacher_logits.shape}")
                    
                logger.info(f"DEBUG: Getting student predictions for step {step}...")
                student_logits, _ = student_model(inputs, targets)
                logger.info(f"DEBUG: Student predictions obtained for step {step}, shape: {student_logits.shape}")
                
                logger.info(f"DEBUG: Computing distillation loss for step {step}...")
                # Move teacher logits back to GPU for loss computation if needed
                if use_memory_efficient:
                    teacher_logits = teacher_logits.to(self.device)
                
                # Compute distillation loss
                loss, loss_components = self.distillation_loss(
                    student_logits, teacher_logits, targets
                )
                
                logger.info(f"DEBUG: Distillation loss computed for step {step}")
                
                # Backward pass (same as base training)
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                
                optimizer.step()
                
                # Store metrics for manager to log
                step_metrics = {
                    "step": step,
                    "train_loss": loss.item(),
                    "distillation_loss": loss_components["distillation_loss"],
                    "ground_truth_loss": loss_components["ground_truth_loss"],
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "gpu_memory_gb": get_gpu_memory_usage()
                }
                training_metrics.append(step_metrics)
                
                # Clear intermediate tensors to save memory
                del inputs, targets, teacher_logits, student_logits, loss
                if use_memory_efficient:
                    clear_gpu_cache()
                    if step % 10 == 0:  # Log memory usage every 10 steps
                        log_memory_usage(step, "after training step")
                
                # Validation check every 100 steps
                if step % 100 == 0:
                    logger.info(f"DEBUG: Running validation for step {step}...")
                    val_loss = self._evaluate_student(student_model, val_dataloader)
                    logger.info(f"DEBUG: Validation completed for step {step}")
                    
                    step_metrics["val_loss"] = val_loss
                    
                    logger.info(f"Step {step}: Train Loss = {step_metrics['train_loss']:.4f}, Val Loss = {val_loss:.4f}, GPU Memory = {step_metrics['gpu_memory_gb']:.2f}GB")
                    
                    # Early stopping and model saving
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best model
                        save_path = os.path.join(save_dir, f"{student_name}_best.pt")
                        
                        # Get student architecture info safely from SimpleNamespace
                        student_architectures = getattr(self.config, 'student_architectures', None)
                        if student_architectures and hasattr(student_architectures, student_name):
                            architecture_info = getattr(student_architectures, student_name, {})
                        else:
                            architecture_info = {}
                        
                        self._save_student_model(student_model, save_path, {
                            "step": step,
                            "val_loss": val_loss,
                            "student_name": student_name,
                            "architecture": architecture_info
                        })
                        
                        logger.info(f"New best validation loss: {val_loss:.4f} at step {step}")
                    else:
                        patience_counter += 1
                        
                    # Update learning rate
                    scheduler.step(val_loss)
                    
                    # Early stopping check
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping triggered after {step} steps")
                        break
                        
        except Exception as e:
            logger.error(f"DEBUG: Error occurred in distillation: {str(e)}")
            logger.error(f"DEBUG: Error type: {type(e)}")
            import traceback
            logger.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
            raise
                
        # Training completed
        training_results = {
            "final_step": step,
            "best_val_loss": best_val_loss,
            "final_epoch": step // 100,  # Approximate epochs based on steps
            "student_name": student_name,
            "compression_ratio": sum(p.numel() for p in self.teacher_model.parameters()) / 
                               sum(p.numel() for p in student_model.parameters()),
            "training_metrics": training_metrics  # For manager to log to W&B
        }
        
        logger.info(f"Distillation completed for {student_name}: {training_results}")
        
        # Clean up training artifacts
        cleanup_training_artifacts(save_dir, keep_best=True)
        
        return training_results
        
    def _evaluate_student(self, student_model: nn.Module, val_dataloader) -> float:
        """
        Evaluate student model on validation set.
        
        Args:
            student_model: Student model to evaluate
            val_dataloader: Validation data loader function
            
        Returns:
            Average validation loss
        """
        student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Create validation data generator
        val_data_gen = val_dataloader()
        
        with torch.no_grad():
            # Evaluate on a limited number of validation batches
            for _ in range(10):  # Evaluate on 10 batches for efficiency
                try:
                    inputs, targets = next(val_data_gen)
                    # Ensure inputs are long type for embedding layers
                    inputs, targets = inputs.to(self.device).long(), targets.to(self.device).long()
                    
                    # Get student predictions
                    student_logits, _ = student_model(inputs, targets)
                    
                    # Compute standard cross-entropy loss for evaluation
                    loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        targets.view(-1)
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clean up for memory efficiency
                    del inputs, targets, student_logits, loss
                    clear_gpu_cache()
                    
                except StopIteration:
                    break
                
        student_model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def _save_student_model(self, model: nn.Module, save_path: str, 
                           metadata: Dict[str, Any]) -> None:
        """
        Save student model checkpoint with metadata.
        
        Args:
            model: Student model to save
            save_path: Path to save the checkpoint
            metadata: Additional metadata to include
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "distillation_config": {
                "temperature": self.config.temperature,
                "alpha": self.config.alpha
            }
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved student model checkpoint to {save_path}")
        
        # W&B upload will be handled by ModelVariantsManager
        
    def distill_all_students(self, train_dataloader, val_dataloader, 
                           save_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Perform distillation for all student architectures.
        
        Args:
            train_dataloader: Training data loader  
            val_dataloader: Validation data loader
            save_dir: Directory to save distilled models
            
        Returns:
            Dictionary mapping student names to their training results
        """
        logger.info("Starting distillation for all student architectures...")
        
        results = {}
        
        for student_name in self.student_models.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"DISTILLING {student_name.upper()} STUDENT MODEL")
            logger.info(f"{'='*60}")
            
            try:
                result = self.distill_student(
                    student_name, train_dataloader, val_dataloader, save_dir
                )
                results[student_name] = result
                
            except Exception as e:
                logger.error(f"Failed to distill {student_name} student: {str(e)}")
                results[student_name] = {"error": str(e)}
                
        logger.info("\nDistillation completed for all students!")
        return results 

    def distill_student_with_quantization(self, student_name: str, train_dataloader, 
                                         val_dataloader, save_dir: str, 
                                         quantization_config, combined_config=None) -> Dict[str, Any]:
        """
        Perform quantization-aware knowledge distillation.
        
        This method trains a student model with quantization simulation,
        allowing it to learn both from the teacher and adapt to quantization
        noise simultaneously.
        
        Args:
            student_name: Name of the student architecture
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            save_dir: Directory to save the distilled model
            quantization_config: Quantization configuration
            combined_config: Combined optimization config (for QAD-specific parameters)
            
        Returns:
            Dictionary containing training metrics and results
        """
        logger.info(f"Starting quantization-aware distillation for {student_name} student model...")
        
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded. Call load_teacher_from_wandb first.")
            
        if student_name not in self.student_models:
            raise ValueError(f"Student model '{student_name}' not found.")
            
        student_model = self.student_models[student_name]
        
        # Prepare student model for quantization-aware training
        student_model = self._prepare_model_for_qat(student_model, quantization_config)
        
        # Use combined config if provided, otherwise fall back to distillation config
        config_to_use = combined_config if combined_config else self.config
        
        # Create QAD-specific distillation loss if combined config is provided
        if combined_config:
            qad_loss = DistillationLoss(
                temperature=combined_config.temperature,
                alpha=combined_config.alpha
            )
        else:
            qad_loss = self.distillation_loss
        
        # Initialize optimizer and scheduler for student using appropriate config
        optimizer = self._create_optimizer(student_model, config_to_use)
        scheduler = self._create_scheduler(optimizer, config_to_use)
        
        # Training metrics tracking
        best_val_loss = float('inf')
        patience_counter = 0
        step = 0
        training_metrics = []
        
        # Step-based training loop (not epoch-based)
        max_steps = 1000  # Reasonable limit for distillation
        
        # Create data generator from the loader function
        train_data_gen = train_dataloader()
        
        for step in range(max_steps):
            student_model.train()
            
            # Get a batch from the generator
            try:
                inputs, targets = next(train_data_gen)
            except StopIteration:
                # If we've exhausted the dataloader, break
                logger.info(f"Exhausted training data at step {step}")
                break
                
            # Ensure inputs are long type for embedding layers
            inputs, targets = inputs.to(self.device).long(), targets.to(self.device).long()
            
            # Get predictions from both models
            with torch.no_grad():
                teacher_logits, _ = self.teacher_model(inputs)
                
            student_logits, _ = student_model(inputs)
            
            # Compute distillation loss
            loss, loss_components = qad_loss(
                student_logits, teacher_logits, targets
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            
            optimizer.step()
            
            # Store metrics for manager to log
            step_metrics = {
                "step": step,
                "train_loss": loss.item(),
                "distillation_loss": loss_components["distillation_loss"],
                "ground_truth_loss": loss_components["ground_truth_loss"],
                "learning_rate": optimizer.param_groups[0]['lr'],
                "qat_enabled": True,
                "quantization_backend": quantization_config.get('backend', 'fbgemm')
            }
            training_metrics.append(step_metrics)
            
            # Validation check every 100 steps
            if step % 100 == 0:
                val_loss = self._evaluate_student(student_model, val_dataloader)
                
                step_metrics["val_loss"] = val_loss
                
                logger.info(f"Step {step}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping and model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model (convert to quantized first)
                    quantized_model = self._convert_qat_to_quantized(student_model)
                    save_path = os.path.join(save_dir, f"{student_name}_qad_best.pt")
                    
                    # Get student architecture info safely from SimpleNamespace
                    student_architectures = getattr(self.config, 'student_architectures', None)
                    if student_architectures and hasattr(student_architectures, student_name):
                        architecture_info = getattr(student_architectures, student_name, {})
                    else:
                        architecture_info = {}
                    
                    self._save_student_model(quantized_model, save_path, {
                        "step": step,
                        "val_loss": val_loss,
                        "student_name": student_name,
                        "architecture": architecture_info,
                        "quantization_aware": True,
                        "backend": quantization_config.get('backend', 'fbgemm')
                    })
                    
                    logger.info(f"New best QAD validation loss: {val_loss:.4f} at step {step}")
                else:
                    patience_counter += 1
                    
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping check
                if patience_counter >= config_to_use.patience:
                    logger.info(f"Early stopping triggered after {step} steps")
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
            "compression_ratio": sum(p.numel() for p in self.teacher_model.parameters()) / 
                               sum(p.numel() for p in final_quantized_model.parameters()),
            "model": final_quantized_model,  # Return the quantized model
            "training_metrics": training_metrics  # For manager to log to W&B
        }
        
        logger.info(f"Quantization-aware distillation completed for {student_name}: {training_results}")
        
        # Clean up training artifacts
        cleanup_training_artifacts(save_dir, keep_best=True)
        
        return training_results
    
    def _prepare_model_for_qat(self, model: nn.Module, quantization_config) -> nn.Module:
        """
        Prepare a model for quantization-aware training.
        
        Args:
            model: Model to prepare
            quantization_config: Quantization configuration
            
        Returns:
            Model prepared for QAT
        """
        # Set quantization configuration
        backend = quantization_config.get('backend', 'fbgemm')
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        
        # Prepare for QAT
        torch.quantization.prepare_qat(model, inplace=True)
        
        logger.info(f"Model prepared for quantization-aware training with backend: {backend}")
        return model
    
    def _convert_qat_to_quantized(self, qat_model: nn.Module) -> nn.Module:
        """
        Convert a QAT model to a fully quantized model.
        
        Args:
            qat_model: Model trained with QAT
            
        Returns:
            Fully quantized model
        """
        qat_model.eval()
        quantized_model = torch.quantization.convert(qat_model, inplace=False)
        logger.info("Converted QAT model to fully quantized model")
        return quantized_model
    
    def distill_all_students_with_quantization(self, train_dataloader, val_dataloader, 
                                             save_dir: str, quantization_config) -> Dict[str, Dict[str, Any]]:
        """
        Perform quantization-aware distillation for all student architectures.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            save_dir: Directory to save distilled models
            quantization_config: Quantization configuration
            
        Returns:
            Dictionary containing results for each student
        """
        logger.info("Starting quantization-aware distillation for all students...")
        
        if not self.student_models:
            raise ValueError("No student models created. Call create_student_models first.")
            
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        for student_name in self.student_models.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"QAD Training: {student_name.upper()} Student")
            logger.info(f"{'='*50}")
            
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

    def _create_optimizer(self, model: nn.Module, config_section) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            model: Model to optimize
            config_section: Configuration section (distillation or combined)
            
        Returns:
            Configured optimizer
        """
        optimizer_config = getattr(config_section, 'optimizer', None)
        
        if optimizer_config and hasattr(optimizer_config, 'type'):
            # Use detailed optimizer config
            if optimizer_config.type == "AdamW":
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config_section.learning_rate,
                    weight_decay=getattr(optimizer_config, 'weight_decay', 0.01),
                    betas=getattr(optimizer_config, 'betas', [0.9, 0.999]),
                    eps=getattr(optimizer_config, 'eps', 1e-8)
                )
            else:
                # Fallback to AdamW with basic config
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config_section.learning_rate,
                    weight_decay=0.01
                )
        else:
            # Fallback to basic AdamW
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config_section.learning_rate,
                weight_decay=0.01
            )
            
        return optimizer
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, config_section) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer to schedule
            config_section: Configuration section (distillation or combined)
            
        Returns:
            Configured scheduler
        """
        scheduler_config = getattr(config_section, 'scheduler', None)
        
        if scheduler_config and hasattr(scheduler_config, 'type'):
            if scheduler_config.type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=getattr(scheduler_config, 'mode', 'min'),
                    factor=getattr(scheduler_config, 'factor', 0.5),
                    patience=getattr(scheduler_config, 'patience', 5),
                    verbose=getattr(scheduler_config, 'verbose', True)
                )
            else:
                # Fallback to ReduceLROnPlateau
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, verbose=True
                )
        else:
            # Fallback scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
        return scheduler 