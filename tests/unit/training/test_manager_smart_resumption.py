"""
Unit tests for smart resumption logic in training manager.

Tests the new functionality for intelligent run ID generation, checkpoint-based
resumption detection, and enhanced metadata tracking.
"""

import pytest
import os
import tempfile
import shutil
import json
from types import SimpleNamespace
from datetime import datetime

from src.training.manager import (
    generate_smart_run_id,
    add_run_metadata,
    save_completion_status,
    load_completion_status
)


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


@pytest.fixture
def sample_checkpoints(mock_config):
    """Create sample checkpoint files for testing."""
    def _create_checkpoint(run_id):
        """Helper to create mock checkpoint files."""
        bs = mock_config.BATCH_SIZE
        cw = mock_config.CONTEXT_WINDOW
        lr = f"{mock_config.LEARNING_RATE:.0e}"
        
        base_filename = f"run_{run_id}_bs{bs}_cw{cw}_lr{lr}"
        latest_path = os.path.join(mock_config.MODEL_CHECKPOINTS_DIR, f"{base_filename}_latest.pt")
        best_path = os.path.join(mock_config.MODEL_CHECKPOINTS_DIR, f"{base_filename}_best.pt")
        
        # Create checkpoint files
        os.makedirs(mock_config.MODEL_CHECKPOINTS_DIR, exist_ok=True)
        with open(latest_path, 'w') as f:
            f.write("mock_checkpoint_data")
        with open(best_path, 'w') as f:
            f.write("mock_checkpoint_data")
            
        return latest_path, best_path
    
    return _create_checkpoint


class TestSmartRunIdGeneration:
    """Test the smart run ID generation logic."""
    
    @pytest.mark.unit
    def test_fresh_start_uses_base_run_id(self, mock_config):
        """Test that fresh start (no checkpoints, no completion status) uses base run ID."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        assert run_id == base_run_id
        assert is_resuming is False
    
    @pytest.mark.unit
    def test_interrupted_run_resumes_with_same_id(self, mock_config, sample_checkpoints):
        """Test that interrupted run (has checkpoints, no completion status) resumes with same ID."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Create checkpoints for base run (simulating interruption)
        sample_checkpoints(base_run_id)
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        assert run_id == base_run_id
        assert is_resuming is True
    
    @pytest.mark.unit
    def test_completed_run_creates_new_suffix(self, mock_config, sample_checkpoints):
        """Test that completed run creates new run with _2 suffix."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Create checkpoints for base run
        sample_checkpoints(base_run_id)
        
        # Mark base run as completed
        completion_info = {"reason": "training_completed", "final_step": 100}
        save_completion_status(base_run_id, mock_config, completion_info)
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        assert run_id == f"{base_run_id}_2"
        assert is_resuming is False
    
    @pytest.mark.unit
    def test_interrupted_suffix_run_resumes_correctly(self, mock_config, sample_checkpoints):
        """Test that interrupted _2 run resumes with _2 ID."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        suffix_run_id = f"{base_run_id}_2"
        
        # Mark base run as completed (so we skip it)
        completion_info = {"reason": "training_completed", "final_step": 100}
        save_completion_status(base_run_id, mock_config, completion_info)
        
        # Create checkpoints for _2 run (simulating interruption of _2)
        sample_checkpoints(suffix_run_id)
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        assert run_id == suffix_run_id
        assert is_resuming is True
    
    @pytest.mark.unit
    def test_multiple_completed_runs_finds_next_available(self, mock_config, sample_checkpoints):
        """Test finding next available suffix when multiple runs are completed."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Mark base run and _2 as completed
        for run_id in [base_run_id, f"{base_run_id}_2"]:
            completion_info = {"reason": "training_completed", "final_step": 100}
            save_completion_status(run_id, mock_config, completion_info)
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        assert run_id == f"{base_run_id}_3"
        assert is_resuming is False
    
    @pytest.mark.unit
    def test_mixed_completed_and_interrupted_runs(self, mock_config, sample_checkpoints):
        """Test behavior with mix of completed and interrupted runs."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Base run completed
        completion_info = {"reason": "training_completed", "final_step": 100}
        save_completion_status(base_run_id, mock_config, completion_info)
        
        # _2 run completed
        run_2_id = f"{base_run_id}_2"
        save_completion_status(run_2_id, mock_config, completion_info)
        
        # _3 run interrupted (has checkpoints but no completion status)
        run_3_id = f"{base_run_id}_3"
        sample_checkpoints(run_3_id)
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        assert run_id == run_3_id
        assert is_resuming is True
    
    @pytest.mark.unit
    def test_completion_tracking_integration(self, mock_config, sample_checkpoints):
        """Test that completion status takes precedence over checkpoint existence."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Create checkpoints AND mark as completed (cleanup failed scenario)
        sample_checkpoints(base_run_id)
        completion_info = {"reason": "training_completed", "final_step": 100}
        save_completion_status(base_run_id, mock_config, completion_info)
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        # Should create new run, not resume (completion status takes precedence)
        assert run_id == f"{base_run_id}_2"
        assert is_resuming is False

    @pytest.mark.unit
    def test_suffix_overflow_uses_timestamp_fallback(self, mock_config, sample_checkpoints):
        """Test that high suffix numbers fall back to timestamp."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Mark base run as completed
        completion_info = {"reason": "training_completed", "final_step": 100}
        save_completion_status(base_run_id, mock_config, completion_info)

        # Mark many suffix runs as completed to trigger timestamp fallback
        for i in range(2, 1002):  # Mark runs _2 through _1001 as completed
            suffix_run_id = f"{base_run_id}_{i}"
            save_completion_status(suffix_run_id, mock_config, completion_info)

        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)

        # Should use timestamp fallback since suffixes 2-1001 are all completed
        assert base_run_id in run_id
        assert len(run_id) > len(base_run_id)  # Should have timestamp suffix
        assert is_resuming is False


class TestRunMetadata:
    """Test the enhanced run metadata functionality."""
    
    @pytest.mark.unit
    def test_add_run_metadata_for_fresh_run(self):
        """Test metadata addition for fresh (non-resuming) run."""
        params = {"test_param": "value"}
        
        enhanced_params = add_run_metadata(params, is_resuming=False)
        
        assert "run_metadata" in enhanced_params
        metadata = enhanced_params["run_metadata"]
        
        assert "created_timestamp" in metadata
        assert "created_readable" in metadata
        assert metadata["is_resuming"] is False
        assert "python_version" in metadata
        
        # Verify timestamp format
        timestamp = datetime.fromisoformat(metadata["created_timestamp"])
        assert isinstance(timestamp, datetime)
    
    @pytest.mark.unit
    def test_add_run_metadata_for_resumed_run(self):
        """Test metadata addition for resumed run."""
        params = {"test_param": "value"}
        
        enhanced_params = add_run_metadata(params, is_resuming=True)
        
        metadata = enhanced_params["run_metadata"]
        assert metadata["is_resuming"] is True
    
    @pytest.mark.unit
    def test_add_run_metadata_preserves_existing_params(self):
        """Test that metadata addition preserves existing parameters."""
        original_params = {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "model_type": "transformer"
        }
        
        enhanced_params = add_run_metadata(original_params.copy(), is_resuming=False)
        
        # Original params should be preserved
        for key, value in original_params.items():
            assert enhanced_params[key] == value
        
        # Metadata should be added
        assert "run_metadata" in enhanced_params


class TestCheckpointDetection:
    """Test checkpoint detection logic within smart run ID generation."""
    
    @pytest.mark.unit
    def test_nonexistent_checkpoint_directory(self):
        """Test behavior when checkpoint directory doesn't exist."""
        config = SimpleNamespace(
            MODEL_CHECKPOINTS_DIR="/nonexistent/directory",
            BATCH_SIZE=16,
            CONTEXT_WINDOW=128,
            LEARNING_RATE=1e-4
        )
        
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Should not crash and should create fresh run (no checkpoints, no completion)
        run_id, is_resuming = generate_smart_run_id(base_run_id, config)
        
        assert run_id == base_run_id
        assert is_resuming is False
    
    @pytest.mark.unit
    def test_empty_checkpoint_directory(self, mock_config):
        """Test behavior with empty checkpoint directory."""
        base_run_id = "single_bs16_cw128_lr1e-04"
        
        # Ensure directory exists but is empty
        os.makedirs(mock_config.MODEL_CHECKPOINTS_DIR, exist_ok=True)
        
        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)
        
        assert run_id == base_run_id
        assert is_resuming is False


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.mark.unit
    def test_malformed_checkpoint_filenames_ignored(self, mock_config):
        """Test that malformed checkpoint files don't interfere."""
        base_run_id = "single_bs16_cw128_lr1e-04"

        # Create some malformed files that shouldn't be detected
        os.makedirs(mock_config.MODEL_CHECKPOINTS_DIR, exist_ok=True)
        malformed_files = [
            "run_malformed_latest.pt",
            "not_a_checkpoint.txt",
            "run_single_bs16_cw128_lr1e-04_best.txt",  # Wrong extension
        ]

        for filename in malformed_files:
            filepath = os.path.join(mock_config.MODEL_CHECKPOINTS_DIR, filename)
            with open(filepath, 'w') as f:
                f.write("malformed")

        run_id, is_resuming = generate_smart_run_id(base_run_id, mock_config)

        # Should ignore malformed files and start fresh
        assert run_id == base_run_id
        assert is_resuming is False 