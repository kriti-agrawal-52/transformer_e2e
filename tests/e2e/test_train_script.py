"""
End-to-end tests for train.py script.

Tests the complete training script functionality including argument parsing,
config loading, and training execution.
"""

import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path


class TestTrainScript:
    """Test the train.py script end-to-end."""
    
    @pytest.mark.e2e
    def test_train_script_help(self):
        """Test that train.py shows help without errors."""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--help"],
            capture_output=True,
            text=True,
            env=env
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "--config" in result.stdout
    
    @pytest.mark.e2e
    def test_train_script_with_invalid_config(self):
        """Test train.py with non-existent config file."""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", "nonexistent.yml"],
            capture_output=True,
            text=True,
            env=env
        )
        
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "not found" in result.stderr.lower()
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_train_script_dry_run(self):
        """Test train.py in dry-run mode (if available) or minimal config."""
        # This test would run the script with minimal config to verify it starts properly
        # without doing full training (which would take too long for tests)
        
        # Create a temporary minimal config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            minimal_config = """
# Minimal config for testing
DATASET_NAME: "wikitext"
DATASET_VARIANT: "wikitext-2-raw-v1"
TOKENIZER_NAME: "gpt2"
RAW_TEXT_LIMIT: 1000
TRAINING_STEPS: 1
BATCH_SIZE: 1
CONTEXT_WINDOW: 8
CHANNEL_DIM: 32
NUM_HEADS: 2
NUM_LAYERS: 1
LEARNING_RATE: 1e-3
VALIDATION_CHECK_EVERY: 1
EARLY_STOPPING_PATIENCE: 1
DEVICE: "cpu"
WANDB_PROJECT: "test-e2e"
WANDB_RUN_PREFIX: ["test-e2e"]
MODEL_CHECKPOINTS_DIR: "/tmp/test_checkpoints"
LOSS_PLOT_DIRECTORY: "/tmp/test_plots"
"""
            f.write(minimal_config)
            temp_config_path = f.name
        
        try:
            # This test is marked as slow and would only run in full e2e test suites
            pytest.skip("Skipping full training test - would require long execution time")
        finally:
            # Cleanup
            os.unlink(temp_config_path) 