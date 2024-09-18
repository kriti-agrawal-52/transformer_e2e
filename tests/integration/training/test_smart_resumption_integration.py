"""
Integration tests for smart resumption logic.

Tests the complete workflow of training interruption, resumption, and 
completion with actual checkpoint files and W&B integration.
"""

import pytest
import torch
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from src.training.manager import run_single_training, TrainingManager


@pytest.fixture
def resumption_test_config():
    """Create a test configuration optimized for resumption testing."""
    temp_dir = tempfile.mkdtemp()
    config = SimpleNamespace(
        # Fast training parameters for testing
        TRAINING_STEPS=20,
        VALIDATION_CHECK_EVERY=5,
        EARLY_STOPPING_PATIENCE=2,
        MIN_DELTA=0.001,
        EVAL_ITERS_VAL=3,
        EVAL_ITERS_TRAIN=2,
        EVAL_ITERS_TEST=3,
        MIN_SUCCESSFUL_VAL_BATCH_RATIO=0.5,
        
        # Small model for fast testing
        BATCH_SIZE=4,
        CONTEXT_WINDOW=16,
        CHANNEL_DIM=32,
        NUM_HEADS=4,
        NUM_LAYERS=2,
        LEARNING_RATE=1e-3,
        DROPOUT_RATE=0.1,
        FINAL_DROPOUT_MULTIPLIER=1.2,
        MAX_DROPOUT_VAL=0.3,
        
        # Infrastructure
        DEVICE="cpu",
        WANDB_PROJECT="test-smart-resumption",
        WANDB_RUN_PREFIX=["test-run", "test-tune"],
        MODEL_CHECKPOINTS_DIR=os.path.join(temp_dir, "checkpoints"),
        LOSS_PLOT_DIRECTORY=os.path.join(temp_dir, "plots"),
        DELETE_LATEST_CHECKPOINT_ON_COMPLETION=True,
        
        # Data
        DATASET_NAME="test-dataset",
    )
    
    # Create directories
    os.makedirs(config.MODEL_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOSS_PLOT_DIRECTORY, exist_ok=True)
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_tokenizer():
    """Create a realistic mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 1000  # Small vocab for testing
    return tokenizer


@pytest.fixture
def sample_text_data():
    """Create sample text data for training."""
    return "This is sample training data. " * 500  # Repeat to have enough data


class TestCompleteResumptionWorkflow:
    """Test the complete smart resumption workflow."""
    
    @pytest.mark.integration
    def test_fresh_run_creates_base_run_id(self, resumption_test_config, mock_tokenizer, sample_text_data):
        """Test that fresh run creates base run ID and completes successfully."""
        config = resumption_test_config
        
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.run") as mock_wandb_run, \
             patch("src.training.manager.log_artifact") as mock_log_artifact:
            
            # Setup W&B mocks
            mock_run = MagicMock()
            mock_run.id = "fresh-run-test"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "fresh-run-test"
            
            # Run training
            run_single_training(sample_text_data, mock_tokenizer, config)
            
            # Verify W&B was initialized with correct metadata
            mock_wandb_init.assert_called_once()
            init_call = mock_wandb_init.call_args
            
            # Check that run_metadata was included
            assert "config" in init_call.kwargs
            config_data = init_call.kwargs["config"]
            assert "run_metadata" in config_data
            
            metadata = config_data["run_metadata"]
            assert metadata["is_resuming"] is False
            assert "created_timestamp" in metadata
            
            # Verify training completed
            mock_wandb_finish.assert_called_once()
    
    @pytest.mark.integration
    def test_interrupted_run_resumption_workflow(self, resumption_test_config, mock_tokenizer, sample_text_data):
        """Test complete workflow: run → interrupt → resume → complete."""
        config = resumption_test_config
        
        # Step 1: Start a training run and "interrupt" it by creating checkpoints
        base_run_id = f"single_bs{config.BATCH_SIZE}_cw{config.CONTEXT_WINDOW}_lr{config.LEARNING_RATE:.0e}"
        
        # Manually create checkpoint files to simulate interrupted run
        base_filename = f"run_{base_run_id}_bs{config.BATCH_SIZE}_cw{config.CONTEXT_WINDOW}_lr{config.LEARNING_RATE:.0e}"
        latest_checkpoint = os.path.join(config.MODEL_CHECKPOINTS_DIR, f"{base_filename}_latest.pt")
        best_checkpoint = os.path.join(config.MODEL_CHECKPOINTS_DIR, f"{base_filename}_best.pt")
        
        # Create realistic checkpoint data
        checkpoint_data = {
            "model_state_dict": {"layer.weight": torch.randn(10, 10)},
            "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": config.LEARNING_RATE}]},
            "step": 10,
            "best_val_loss": 2.5,
            "completed": False
        }
        
        torch.save(checkpoint_data, latest_checkpoint)
        torch.save(checkpoint_data, best_checkpoint)
        
        # Step 2: Restart training - should resume with same run ID
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.run") as mock_wandb_run, \
             patch("src.training.manager.log_artifact") as mock_log_artifact:
            
            mock_run = MagicMock()
            mock_run.id = f"resume-{base_run_id}"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = f"resume-{base_run_id}"
            
            # Run training (should resume)
            run_single_training(sample_text_data, mock_tokenizer, config)
            
            # Verify W&B was initialized with resumption metadata
            mock_wandb_init.assert_called_once()
            init_call = mock_wandb_init.call_args
            
            config_data = init_call.kwargs["config"]
            metadata = config_data["run_metadata"]
            assert metadata["is_resuming"] is True
            
            # Verify completion
            mock_wandb_finish.assert_called_once()
    
    @pytest.mark.integration
    def test_completed_run_creates_new_run_with_suffix(self, resumption_test_config, mock_tokenizer, sample_text_data):
        """Test that completed run (no checkpoints) creates new run with _2 suffix."""
        config = resumption_test_config
        
        # Step 1: Simulate that base run was completed by NOT having checkpoints
        # (In real scenario, checkpoints would be cleaned up after successful completion)
        
        # Step 2: Start training - should create new run with _2 suffix
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.run") as mock_wandb_run, \
             patch("src.training.manager.log_artifact") as mock_log_artifact:
            
            mock_run = MagicMock()
            mock_run.id = "new-run-with-suffix"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "new-run-with-suffix"
            
            # Run training
            run_single_training(sample_text_data, mock_tokenizer, config)
            
            # Verify W&B was initialized with correct metadata
            mock_wandb_init.assert_called_once()
            init_call = mock_wandb_init.call_args
            
            # Check run ID in config
            config_data = init_call.kwargs["config"]
            assert "_2" in config_data["run_id"]  # Should have _2 suffix
            
            metadata = config_data["run_metadata"]
            assert metadata["is_resuming"] is False
    
    @pytest.mark.integration
    def test_checkpoint_cleanup_after_successful_completion(self, resumption_test_config, mock_tokenizer, sample_text_data):
        """Test that checkpoints are cleaned up after successful completion."""
        config = resumption_test_config
        
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.run") as mock_wandb_run, \
             patch("src.training.manager.log_artifact") as mock_log_artifact, \
             patch("src.training.manager.wait_for_artifact_upload", return_value=True):
            
            mock_run = MagicMock()
            mock_run.id = "cleanup-test"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "cleanup-test"
            
            # Run training to completion
            run_single_training(sample_text_data, mock_tokenizer, config)
            
            # Check that no checkpoint files remain (they should be cleaned up)
            checkpoint_files = [f for f in os.listdir(config.MODEL_CHECKPOINTS_DIR) 
                              if f.endswith(".pt")]
            
            # After successful completion, checkpoints should be cleaned up
            # (This enables the smart resumption logic to work correctly)
            assert len(checkpoint_files) == 0, f"Checkpoints should be cleaned up after completion, but found: {checkpoint_files}"


class TestResumptionEdgeCases:
    """Test edge cases in the resumption workflow."""
    
    @pytest.mark.integration
    def test_corrupted_checkpoint_handles_gracefully(self, resumption_test_config, mock_tokenizer, sample_text_data):
        """Test that corrupted checkpoints are handled gracefully."""
        config = resumption_test_config
        
        # Create corrupted checkpoint file
        base_run_id = f"single_bs{config.BATCH_SIZE}_cw{config.CONTEXT_WINDOW}_lr{config.LEARNING_RATE:.0e}"
        base_filename = f"run_{base_run_id}_bs{config.BATCH_SIZE}_cw{config.CONTEXT_WINDOW}_lr{config.LEARNING_RATE:.0e}"
        corrupted_checkpoint = os.path.join(config.MODEL_CHECKPOINTS_DIR, f"{base_filename}_latest.pt")
        
        # Write corrupted data
        with open(corrupted_checkpoint, 'w') as f:
            f.write("corrupted_checkpoint_data")
        
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.run") as mock_wandb_run:
            
            mock_run = MagicMock()
            mock_run.id = "corrupted-test"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "corrupted-test"
            
            # Should not crash even with corrupted checkpoint
            try:
                run_single_training(sample_text_data, mock_tokenizer, config)
                # If it doesn't crash, the test passes
                assert True
            except Exception as e:
                # Should handle corruption gracefully
                assert "corrupted" in str(e).lower() or "invalid" in str(e).lower()
    
    @pytest.mark.integration
    def test_permission_error_during_cleanup_does_not_fail_run(self, resumption_test_config, mock_tokenizer, sample_text_data):
        """Test that permission errors during cleanup don't fail the run."""
        config = resumption_test_config
        
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.run") as mock_wandb_run, \
             patch("src.training.manager.log_artifact") as mock_log_artifact, \
             patch("os.remove") as mock_remove:
            
            # Mock permission error during cleanup
            mock_remove.side_effect = PermissionError("Permission denied")
            
            mock_run = MagicMock()
            mock_run.id = "permission-test"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "permission-test"
            
            # Should complete successfully despite cleanup failure
            try:
                run_single_training(sample_text_data, mock_tokenizer, config)
                assert True  # Should not crash
            except PermissionError:
                pytest.fail("Permission error during cleanup should be handled gracefully")


class TestMetadataIntegration:
    """Test W&B metadata integration with smart resumption."""
    
    @pytest.mark.integration
    def test_wandb_receives_complete_metadata(self, resumption_test_config, mock_tokenizer, sample_text_data):
        """Test that W&B receives complete metadata including resumption status."""
        config = resumption_test_config
        
        with patch("wandb.init") as mock_wandb_init, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch("wandb.run") as mock_wandb_run:
            
            mock_run = MagicMock()
            mock_run.id = "metadata-test"
            mock_wandb_init.return_value = mock_run
            mock_wandb_run.id = "metadata-test"
            
            run_single_training(sample_text_data, mock_tokenizer, config)
            
            # Check W&B init call
            mock_wandb_init.assert_called_once()
            init_call = mock_wandb_init.call_args
            
            # Verify complete config was passed
            config_data = init_call.kwargs["config"]
            
            # Check metadata structure
            assert "run_metadata" in config_data
            metadata = config_data["run_metadata"]
            
            required_metadata_keys = [
                "created_timestamp", 
                "created_readable", 
                "is_resuming", 
                "python_version"
            ]
            
            for key in required_metadata_keys:
                assert key in metadata, f"Missing metadata key: {key}"
            
            # Check W&B log calls for additional metadata
            assert mock_wandb_log.call_count > 0
            
            # Look for metadata logging calls
            metadata_logged = False
            for call in mock_wandb_log.call_args_list:
                if len(call.args) > 0 and isinstance(call.args[0], dict):
                    log_data = call.args[0]
                    if any(key.startswith("run_created") or key.startswith("is_resuming") 
                          for key in log_data.keys()):
                        metadata_logged = True
                        break
            
            assert metadata_logged, "Should log additional metadata to W&B" 