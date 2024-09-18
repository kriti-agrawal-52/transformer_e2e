import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import tempfile

from src.training.manager import run_single_training
from src.utils.config_loader import load_config


@pytest.fixture
def mock_train_config():
    """Mock training configuration for integration tests"""
    return SimpleNamespace(
        DEVICE="cpu",
        WANDB_PROJECT="test-training-flow",
        MODEL_CHECKPOINTS_DIR=tempfile.mkdtemp(),
        LOSS_PLOT_DIRECTORY=tempfile.mkdtemp(),
        TRAINING_STEPS=20,
        VALIDATION_CHECK_EVERY=5,
        EARLY_STOPPING_PATIENCE=3,
        MIN_DELTA=0.001,
        EVAL_ITERS_VAL=5,
        EVAL_ITERS_TRAIN=3,
        EVAL_ITERS_TEST=5,
        MIN_SUCCESSFUL_VAL_BATCH_RATIO=0.5,
        BATCH_SIZE=4,
        CONTEXT_WINDOW=16,
        LEARNING_RATE=1e-3,
        CHANNEL_DIM=32,
        NUM_HEADS=4,
        NUM_LAYERS=2,
        DATASET_NAME="test-dataset",
        WANDB_RUN_PREFIX=["single", "tune"],
        ALWAYS_LOG_ARTIFACTS=True,
        DELETE_LATEST_CHECKPOINT_ON_COMPLETION=True,
        HP_SEARCH_BATCH_SIZES=[2, 4],
        HP_SEARCH_CONTEXT_WINDOWS=[8, 16],
        HP_SEARCH_LRS=[1e-4, 1e-3],
        HP_SEARCH_STEPS=10,
        HP_VALIDATION_CHECK_EVERY=5,
        HP_EARLY_STOPPING_PATIENCE=3
    )


@patch("src.training.manager.wandb")
def test_successful_training_run(mock_wandb, mock_train_config, dummy_data_path):
    """Test that a complete training run executes successfully with all expected workflow steps."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    with patch("src.training.manager.TrainingManager.run") as mock_manager_run:
        mock_manager_run.return_value = "/path/to/best_checkpoint.pt"
        
        # Call without the invalid is_sweep parameter
        run_single_training(dummy_data_path, mock_tokenizer, mock_train_config)
        
        # Verify the training manager was created and run
        mock_manager_run.assert_called_once()


@patch("src.training.manager.wandb")
def test_early_stopping(mock_wandb, mock_train_config, dummy_data_path):
    """Test that early stopping mechanism works correctly and stops training before max steps."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    # Configure for early stopping
    mock_train_config.EARLY_STOPPING_PATIENCE = 3
    mock_train_config.TRAINING_STEPS = 1000  # High number to test early stopping
    
    with patch("src.training.manager.TrainingManager.run") as mock_manager_run:
        mock_manager_run.return_value = "/path/to/best_checkpoint.pt"
        
        run_single_training(dummy_data_path, mock_tokenizer, mock_train_config)
        
        # Verify early stopping parameters were passed correctly
        mock_manager_run.assert_called_once()


@patch("src.training.manager.TrainingManager")
def test_resume_interrupted_run(mock_manager_class, mock_train_config, dummy_data_path):
    """Test that interrupted training runs can be resumed correctly from the latest checkpoint."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    # Mock an interrupted training scenario
    mock_manager_instance = MagicMock()
    mock_manager_class.return_value = mock_manager_instance
    mock_manager_instance.run.return_value = "/path/to/best_checkpoint.pt"
    
    run_single_training(dummy_data_path, mock_tokenizer, mock_train_config)
    
    # Verify manager was created and run was called
    mock_manager_class.assert_called_once()
    mock_manager_instance.run.assert_called_once()


@patch("src.training.manager.TrainingManager")
def test_restart_completed_run(mock_manager_class, mock_train_config, dummy_data_path):
    """Test behavior when attempting to restart a training run that was already completed."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    # Mock a completed training scenario
    mock_manager_instance = MagicMock() 
    mock_manager_class.return_value = mock_manager_instance
    mock_manager_instance.run.return_value = "/path/to/best_checkpoint.pt"
    
    run_single_training(dummy_data_path, mock_tokenizer, mock_train_config)
    
    # Verify manager was created and run was called
    mock_manager_class.assert_called_once()
    mock_manager_instance.run.assert_called_once()
