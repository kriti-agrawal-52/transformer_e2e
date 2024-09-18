"""
Unit tests for training utilities.

Tests utility functions like checkpoint path generation, checkpoint operations,
and validation loss computation.
"""

import pytest
import os
import tempfile
import shutil
from types import SimpleNamespace

from src.training.utils import get_checkpoint_paths, save_checkpoint, load_checkpoint


@pytest.fixture
def mock_config():
    """Create a mock configuration with temporary directories."""
    temp_dir = tempfile.mkdtemp()
    config = SimpleNamespace(
        MODEL_CHECKPOINTS_DIR=temp_dir,
        BATCH_SIZE=16,
        CONTEXT_WINDOW=128,
        LEARNING_RATE=1e-4
    )
    yield config
    # Cleanup
    shutil.rmtree(temp_dir)


class TestCheckpointPaths:
    """Test the centralized checkpoint path generation utility."""
    
    @pytest.mark.unit
    def test_checkpoint_paths_with_cfg_only(self, mock_config):
        """Test checkpoint path generation using only cfg parameters."""
        run_id = "test_run"
        
        paths = get_checkpoint_paths(run_id, mock_config)
        
        expected_base = "run_test_run_bs16_cw128_lr1e-04"
        assert paths['base_filename'] == expected_base
        assert paths['latest'].endswith(f"{expected_base}_latest.pt")
        assert paths['best'].endswith(f"{expected_base}_best.pt")
        assert mock_config.MODEL_CHECKPOINTS_DIR in paths['latest']
        assert mock_config.MODEL_CHECKPOINTS_DIR in paths['best']
    
    @pytest.mark.unit
    def test_checkpoint_paths_with_run_params(self, mock_config):
        """Test checkpoint path generation using run_params (overrides cfg)."""
        run_id = "test_run"
        run_params = {
            'batch_size': 32,
            'context_window': 256,
            'learning_rate': 2e-3
        }
        
        paths = get_checkpoint_paths(run_id, mock_config, run_params)
        
        expected_base = "run_test_run_bs32_cw256_lr2e-03"
        assert paths['base_filename'] == expected_base
        assert paths['latest'].endswith(f"{expected_base}_latest.pt")
        assert paths['best'].endswith(f"{expected_base}_best.pt")
    
    @pytest.mark.unit
    def test_checkpoint_paths_learning_rate_formatting(self, mock_config):
        """Test that learning rates are formatted consistently."""
        run_id = "test_run"
        
        # Test with different learning rate formats
        test_cases = [
            ({'batch_size': 16, 'context_window': 128, 'learning_rate': 1e-4}, "lr1e-04"),
            ({'batch_size': 16, 'context_window': 128, 'learning_rate': 0.001}, "lr1e-03"),
            ({'batch_size': 16, 'context_window': 128, 'learning_rate': 0.5}, "lr5e-01"),
        ]
        
        for run_params, expected_lr in test_cases:
            paths = get_checkpoint_paths(run_id, mock_config, run_params)
            assert expected_lr in paths['base_filename']
    
    @pytest.mark.unit
    def test_checkpoint_paths_handles_missing_cfg_attributes(self):
        """Test graceful handling of missing cfg attributes."""
        incomplete_config = SimpleNamespace(MODEL_CHECKPOINTS_DIR="/tmp/test")
        run_id = "test_run"
        
        paths = get_checkpoint_paths(run_id, incomplete_config)
        
        # Should use "unknown" for missing attributes
        assert "unknown" in paths['base_filename']
        assert paths['latest'].endswith("_latest.pt")
        assert paths['best'].endswith("_best.pt")


class TestCheckpointOperations:
    """Test checkpoint save/load operations."""
    
    @pytest.mark.unit
    def test_checkpoint_path_consistency(self, mock_config):
        """Test that checkpoint paths are generated consistently across functions."""
        run_id = "consistency_test"
        run_params = {
            'batch_size': 16,
            'context_window': 128,
            'learning_rate': 1e-4
        }
        
        # Generate paths using the utility
        paths = get_checkpoint_paths(run_id, mock_config, run_params)
        
        # Verify the paths match expected pattern
        expected_pattern = f"run_{run_id}_bs16_cw128_lr1e-04"
        assert expected_pattern in paths['latest']
        assert expected_pattern in paths['best']
        
        # Ensure both paths are in the correct directory
        assert paths['latest'].startswith(mock_config.MODEL_CHECKPOINTS_DIR)
        assert paths['best'].startswith(mock_config.MODEL_CHECKPOINTS_DIR) 