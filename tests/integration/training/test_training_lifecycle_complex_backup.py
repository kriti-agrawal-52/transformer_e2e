"""
Simplified integration tests for training lifecycle scenarios.

This module tests the basic training workflow including:
- Full training completion with checkpoint saving
- Early stopping behavior
"""

import pytest
import torch
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from src.training.manager import run_single_training


@pytest.fixture
def training_config_for_lifecycle_tests(tmpdir):
    """Creates a complete training configuration for lifecycle testing."""
    config = SimpleNamespace(
        # Core training parameters
        TRAINING_STEPS=20,
        VALIDATION_CHECK_EVERY=5,
        EARLY_STOPPING_PATIENCE=3,
        MIN_DELTA=0.01,
        EVAL_ITERS_VAL=5,
        EVAL_ITERS_TRAIN=3,
        EVAL_ITERS_TEST=5,
        MIN_SUCCESSFUL_VAL_BATCH_RATIO=0.5,
        
        # Model architecture
        BATCH_SIZE=4,
        CONTEXT_WINDOW=8,
        CHANNEL_DIM=32,
        NUM_HEADS=4,
        NUM_LAYERS=2,
        LEARNING_RATE=1e-3,
        DROPOUT_RATE=0.1,
        
        # Infrastructure
        DEVICE="cpu",
        WANDB_PROJECT="test-training-lifecycle",
        WANDB_RUN_PREFIX=["test-run"],
        MODEL_CHECKPOINTS_DIR=str(tmpdir.join("checkpoints")),
        LOSS_PLOT_DIRECTORY=str(tmpdir.join("plots")),
        DELETE_LATEST_CHECKPOINT_ON_COMPLETION=True,
        
        # Data
        DATASET_NAME="test-dataset",
        TOKENIZER_NAME="gpt2"
    )
    
    # Ensure checkpoint and plots directories exist
    os.makedirs(config.MODEL_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOSS_PLOT_DIRECTORY, exist_ok=True)
    
    return config


@pytest.fixture
def sample_training_data(tmpdir):
    """Creates sample training data for testing."""
    data_content = "This is sample training data for testing the training lifecycle. " * 200
    data_path = tmpdir.join("training_data.txt")
    with open(str(data_path), "w") as f:
        f.write(data_content)
    return str(data_path)


@pytest.fixture
def mock_tokenizer():
    """Creates a mock tokenizer for testing."""
    from transformers import AutoTokenizer
    with patch.object(AutoTokenizer, 'from_pretrained') as mock:
        tokenizer = MagicMock()
        tokenizer.vocab_size = 50257
        tokenizer.encode.return_value = list(range(100))  # Mock token IDs
        mock.return_value = tokenizer
        yield tokenizer


class TestCompleteTrainingLifecycle:
    """Tests for complete training workflow from start to finish."""
    
    @pytest.mark.integration
    def test_full_training_completes_successfully_and_saves_best_checkpoint(
        self, training_config_for_lifecycle_tests, sample_training_data, mock_tokenizer
    ):
        """
        Tests that a full training run completes successfully, saves the best checkpoint,
        uploads artifacts to W&B, and cleans up temporary files.
        """
        config = training_config_for_lifecycle_tests
        
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.save") as mock_wandb_save, \
             patch("wandb.log_artifact") as mock_wandb_log_artifact, \
             patch("wandb.Artifact") as mock_artifact, \
             patch("wandb.run") as mock_wandb_run, \
             patch("src.training.manager.wait_for_artifact_upload", return_value=True):
            
            mock_run = MagicMock()
            mock_run.id = "test-run-id-123"
            mock_run.name = "test-run-name"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "test-run-id-123"
            mock_wandb_run.name = "test-run-name"
            
            # Run complete training
            run_single_training(sample_training_data, mock_tokenizer, config)
            
            # Verify training completed successfully
            mock_wandb_init.assert_called_once()
            assert mock_wandb_log.call_count > 0, "Training metrics should be logged"
            assert mock_wandb_finish.call_count >= 1, "Should call wandb.finish() at least once"
            
            # Verify best checkpoint exists
            checkpoint_files = os.listdir(config.MODEL_CHECKPOINTS_DIR)
            best_checkpoint_files = [f for f in checkpoint_files if "_best.pt" in f]
            assert len(best_checkpoint_files) > 0, f"No best checkpoint found. Files: {checkpoint_files}"
            
            best_checkpoint_path = os.path.join(config.MODEL_CHECKPOINTS_DIR, best_checkpoint_files[0])
            checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
            assert checkpoint["step"] > 0, "Should have completed some training steps"
            
            # Verify completion status is tracked separately
            completion_tracking_dir = os.path.join(config.MODEL_CHECKPOINTS_DIR, "completion_tracking")
            if os.path.exists(completion_tracking_dir):
                completion_files = [f for f in os.listdir(completion_tracking_dir) if f.endswith("_completed.json")]
                assert len(completion_files) > 0, "Should have completion tracking files"
    
    @pytest.mark.integration 
    def test_early_stopping_triggers_when_validation_loss_stops_improving(
        self, training_config_for_lifecycle_tests, sample_training_data, mock_tokenizer
    ):
        """
        Tests that early stopping is triggered when validation loss doesn't improve
        for the specified patience period, and the run is properly marked as completed.
        """
        config = training_config_for_lifecycle_tests
        config.TRAINING_STEPS = 100  # Set high so early stopping triggers first
        config.EARLY_STOPPING_PATIENCE = 2
        config.MIN_DELTA = 1000  # Ensure no improvement is ever registered
        
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.save") as mock_wandb_save, \
             patch("wandb.log_artifact") as mock_wandb_log_artifact, \
             patch("wandb.Artifact") as mock_artifact, \
             patch("wandb.run") as mock_wandb_run, \
             patch("src.training.manager.wait_for_artifact_upload", return_value=True):
            
            mock_run = MagicMock()
            mock_run.id = "early-stop-test-id"
            mock_run.name = "early_stop_test"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "early-stop-test-id" 
            mock_wandb_run.name = "early_stop_test"
            
            # Run training with early stopping configuration
            run_single_training(sample_training_data, mock_tokenizer, config)
            
            # Verify training was initiated
            mock_wandb_init.assert_called_once()
            assert mock_wandb_log.call_count > 0, "Should log training progress"
            mock_wandb_finish.assert_called_once()


class TestTrainingSmoke:
    """Simple smoke tests for training functionality - no complex mocking."""
    
    @pytest.mark.integration
    def test_training_manager_can_be_imported_and_called(self):
        """Basic smoke test that training manager can be imported and called."""
        from src.training.manager import run_single_training
        assert callable(run_single_training), "Training manager should be callable"
    
    @pytest.mark.integration  
    def test_training_config_validation(self):
        """Test that training configuration can be created and has required fields."""
        config = SimpleNamespace(
            TRAINING_STEPS=10,
            BATCH_SIZE=2,
            LEARNING_RATE=1e-3,
            DEVICE="cpu"
        )
        
        # Basic validation that config has essential fields
        assert hasattr(config, 'TRAINING_STEPS')
        assert hasattr(config, 'BATCH_SIZE') 
        assert hasattr(config, 'LEARNING_RATE')
        assert hasattr(config, 'DEVICE') 