import pytest
import torch
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path to allow direct import
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from training.manager import TrainingManager, ValidationLossComputationError


@pytest.fixture
def mock_training_manager(mock_train_config, test_artifacts_dir):
    """Fixture to create a TrainingManager with mocked components for unit testing."""
    with patch("torch.optim.AdamW") as mock_optim:
        with patch("src.models.transformer.TransformerModel") as mock_model:
            # Mock the data preprocessor to avoid actual data loading/processing
            mock_preprocessor = MagicMock()
            mock_preprocessor.get_batch = MagicMock(
                return_value=(torch.rand(1, 8), torch.rand(1, 8))
            )

            manager = TrainingManager(
                cfg=mock_train_config,
                model=mock_model,
                optimizer=mock_optim,
                data_preprocessor=mock_preprocessor,
                device="cpu",
                run_name="test_run",
            )
            # Mock logger and wandb_run to isolate the manager's logic
            manager.logger = MagicMock()
            manager.wandb_run = MagicMock()
            yield manager


@pytest.mark.unit
def test_get_run_name(mock_train_config):
    """Tests that a run name is generated correctly."""
    name = TrainingManager._get_run_name(mock_train_config)
    assert name.startswith("test-run_")
    assert len(name) > len("test-run_")


@pytest.mark.unit
def test_checkpoint_path_generation(mock_training_manager):
    """Tests the generation of checkpoint paths."""
    manager = mock_training_manager
    latest_path = manager._get_checkpoint_path()
    best_path = manager._get_checkpoint_path(best=True)

    assert "test_run_latest.pt" in latest_path
    assert "test_run_best.pt" in best_path


@pytest.mark.unit
def test_save_and_load_checkpoint(mock_training_manager):
    """Tests saving and loading a checkpoint, verifying the saved state."""
    manager = mock_training_manager
    manager.current_step = 100
    manager.best_val_loss = 0.1

    # Mock the model and optimizer state dicts
    manager.model.state_dict.return_value = {"param": torch.tensor(1.0)}
    manager.optimizer.state_dict.return_value = {"state": "optim_state"}

    checkpoint_path = os.path.join(manager.cfg.MODEL_CHECKPOINTS_DIR, "checkpoint.pt")

    # Test Save
    with patch("torch.save") as mock_torch_save:
        manager._save_checkpoint(checkpoint_path)
        mock_torch_save.assert_called_once()
        saved_data = mock_torch_save.call_args[0][0]
        assert saved_data["step"] == 100
        assert saved_data["best_val_loss"] == 0.1
        assert "model_state_dict" in saved_data

    # Test Load
    with patch("torch.load", return_value=saved_data) as mock_torch_load:
        manager._load_checkpoint(checkpoint_path)
        mock_torch_load.assert_called_once_with(checkpoint_path, map_location="cpu")
        manager.model.load_state_dict.assert_called_once_with(
            {"param": torch.tensor(1.0)}
        )
        manager.optimizer.load_state_dict.assert_called_once_with(
            {"state": "optim_state"}
        )
        assert manager.current_step == 100
        assert manager.best_val_loss == 0.1


@pytest.mark.unit
def test_early_stopping_logic(mock_training_manager):
    """Tests the early stopping logic increments and triggers correctly."""
    manager = mock_training_manager
    manager.patience_counter = 0
    manager.best_val_loss = 1.0

    # Loss improved -> counter should reset
    manager._check_early_stopping(current_val_loss=0.8)
    assert manager.patience_counter == 0
    assert manager.best_val_loss == 0.8

    # Loss did not improve -> counter should increment
    manager._check_early_stopping(current_val_loss=0.85)
    assert manager.patience_counter == 1

    # Patience limit reached -> should trigger stop
    manager.patience_counter = manager.cfg.EARLY_STOPPING_PATIENCE - 1
    stop = manager._check_early_stopping(current_val_loss=0.9)
    assert stop is False

    stop = manager._check_early_stopping(current_val_loss=0.9)
    assert stop is True


@pytest.mark.unit
def test_completed_run_check_prevents_retraining(mock_training_manager):
    """Tests that a 'completed' checkpoint prevents retraining."""
    manager = mock_training_manager

    completed_checkpoint = {"completed": True}
    with patch("torch.load", return_value=completed_checkpoint), patch(
        "os.path.exists", return_value=True
    ):

        manager._resume_if_needed()
        assert manager.is_resumed
        assert manager.is_completed

        # The train method should do nothing if the run is already completed
        manager.train()
        manager.model.forward.assert_not_called()


@pytest.mark.unit
def test_validation_loss_error_when_failing_batches(mock_training_manager):
    """
    Tests that ValidationLossComputationError is raised when the success ratio is too low.
    """
    manager = mock_training_manager
    manager.cfg.MIN_SUCCESSFUL_VAL_BATCH_RATIO = 0.6
    manager.cfg.EVAL_ITERS_VAL = 10

    # Simulate estimate_loss failing on more than 40% of batches
    side_effects = [Exception("Simulated failure")] * 5 + [0.5] * 5
    with patch.object(manager, "estimate_loss", side_effect=side_effects):
        with pytest.raises(ValidationLossComputationError):
            manager._compute_validation_loss()


@pytest.mark.unit
def test_validation_loss_warning_on_some_failures(mock_training_manager):
    """
    Tests that a warning is logged if some (but not too many) validation batches fail.
    """
    manager = mock_training_manager
    manager.cfg.MIN_SUCCESSFUL_VAL_BATCH_RATIO = 0.4
    manager.cfg.EVAL_ITERS_VAL = 10

    # Simulate estimate_loss failing on 50% of batches (which is > 40% success ratio)
    side_effects = [Exception("Simulated failure")] * 5 + [torch.tensor(0.5)] * 5
    with patch.object(manager, "estimate_loss", side_effect=side_effects):
        val_loss = manager._compute_validation_loss()

        # A warning should have been logged about the failed batches
        assert manager.logger.warning.call_count > 0
        assert (
            "5/10 validation batches failed" in manager.logger.warning.call_args[0][0]
        )
        # The loss should be the average of the successful batches
        assert val_loss == 0.5


@pytest.mark.unit
def test_load_corrupted_checkpoint(mock_training_manager):
    """
    Tests that loading a corrupted or incomplete checkpoint raises a KeyError.
    """
    manager = mock_training_manager

    # Create a checkpoint that is missing the 'model_state_dict' key
    corrupted_checkpoint = {
        "step": 100,
        "best_val_loss": 0.1,
        # 'model_state_dict' is missing
    }

    checkpoint_path = os.path.join(manager.cfg.MODEL_CHECKPOINTS_DIR, "corrupted.pt")

    with patch("torch.load", return_value=corrupted_checkpoint):
        with pytest.raises(KeyError):
            manager._load_checkpoint(checkpoint_path)
