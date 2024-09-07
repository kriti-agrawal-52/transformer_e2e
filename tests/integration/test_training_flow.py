import pytest
import torch
import os
from unittest.mock import patch, MagicMock

from src.training.manager import run_single_training
from src.utils.config_loader import load_config


@pytest.fixture
def setup_mocks_for_training():
    """Centralized mock setup for all training integration tests."""
    with patch("wandb.init") as mock_wandb_init, patch(
        "wandb.log"
    ) as mock_wandb_log, patch("wandb.finish") as mock_wandb_finish, patch(
        "wandb.Artifact"
    ) as mock_artifact, patch(
        "src.training.manager.wait_for_artifact_upload", return_value=True
    ) as mock_wait_for_upload:

        # Mock the artifact object to have a name and be usable in a 'with' statement
        mock_artifact_instance = MagicMock()
        mock_artifact_instance.name = "test_artifact"
        mock_artifact.return_value = mock_artifact_instance

        # Mock wandb.run to be able to log artifacts
        mock_run = MagicMock()
        mock_run.log_artifact = MagicMock()
        mock_wandb_init.return_value = mock_run

        yield {
            "init": mock_wandb_init,
            "log": mock_wandb_log,
            "finish": mock_wandb_finish,
            "artifact": mock_artifact,
            "log_artifact": mock_run.log_artifact,
            "wait_for_upload": mock_wait_for_upload,
            "run": mock_run,
        }


@pytest.mark.integration
def test_successful_training_run(
    mock_train_config, dummy_data_path, setup_mocks_for_training
):
    """
    Tests a complete, successful training run, verifying artifact creation,
    logging, and cleanup.
    """
    mock_train_config.TRAINING_STEPS = 5
    mock_train_config.VALIDATION_CHECK_EVERY = 2
    mock_train_config.DELETE_LATEST_CHECKPOINT_ON_COMPLETION = True

    run_single_training(mock_train_config, dummy_data_path, is_sweep=False)

    setup_mocks_for_training["init"].assert_called_once()
    assert setup_mocks_for_training["log"].call_count > 0
    setup_mocks_for_training["artifact"].assert_called()
    setup_mocks_for_training["log_artifact"].assert_called()
    setup_mocks_for_training["wait_for_upload"].assert_called()
    setup_mocks_for_training["finish"].assert_called_once()

    run_name = setup_mocks_for_training["run"].name
    best_checkpoint_path = os.path.join(
        mock_train_config.MODEL_CHECKPOINTS_DIR, f"{run_name}_best.pt"
    )
    assert os.path.exists(best_checkpoint_path)

    checkpoint = torch.load(best_checkpoint_path)
    assert checkpoint["completed"] is True

    latest_checkpoint_path = os.path.join(
        mock_train_config.MODEL_CHECKPOINTS_DIR, f"{run_name}_latest.pt"
    )
    assert not os.path.exists(latest_checkpoint_path)


@pytest.mark.integration
def test_early_stopping(mock_train_config, dummy_data_path, setup_mocks_for_training):
    """
    Tests that early stopping is triggered and the run is marked as completed.
    """
    mock_train_config.TRAINING_STEPS = 50
    mock_train_config.VALIDATION_CHECK_EVERY = 2
    mock_train_config.EARLY_STOPPING_PATIENCE = 2
    mock_train_config.MIN_DELTA = 1000  # Ensure no improvement is ever registered

    run_single_training(mock_train_config, dummy_data_path, is_sweep=False)

    run_name = setup_mocks_for_training["run"].name
    best_checkpoint_path = os.path.join(
        mock_train_config.MODEL_CHECKPOINTS_DIR, f"{run_name}_best.pt"
    )
    checkpoint = torch.load(best_checkpoint_path)

    assert checkpoint["step"] < mock_train_config.TRAINING_STEPS
    assert checkpoint["completed"] is True


@pytest.mark.integration
def test_resume_interrupted_run(
    mock_train_config, dummy_data_path, setup_mocks_for_training
):
    """
    Tests resuming a run that was 'interrupted' mid-way.
    """
    # --- First run (gets interrupted) ---
    mock_train_config.TRAINING_STEPS = 10

    run_name = "resume_test_run"
    with patch(
        "src.training.manager.TrainingManager._get_run_name", return_value=run_name
    ):
        # Patch the training loop to exit early and simulate a crash
        with patch("src.training.manager.TrainingManager.train") as mock_train_method:

            def side_effect(self, *args, **kwargs):
                self.current_step = 3
                self._save_checkpoint(self._get_checkpoint_path())
                raise KeyboardInterrupt

            mock_train_method.side_effect = side_effect

            with pytest.raises(KeyboardInterrupt):
                run_single_training(mock_train_config, dummy_data_path, is_sweep=False)

    # --- Second run (resumes) ---
    setup_mocks_for_training["init"].reset_mock()
    with patch(
        "src.training.manager.TrainingManager._get_run_name", return_value=run_name
    ):
        run_single_training(mock_train_config, dummy_data_path, is_sweep=False)

    final_checkpoint_path = os.path.join(
        mock_train_config.MODEL_CHECKPOINTS_DIR, f"{run_name}_best.pt"
    )
    checkpoint = torch.load(final_checkpoint_path)
    assert checkpoint["step"] == 10
    assert checkpoint["completed"] is True

    # Check that wandb.init was called with resume=True
    setup_mocks_for_training["init"].assert_called_with(
        project=mock_train_config.WANDB_PROJECT,
        config=mock_train_config,
        name=run_name,
        resume="allow",
        id=run_name,
    )


@pytest.mark.integration
def test_restart_completed_run(
    mock_train_config, dummy_data_path, setup_mocks_for_training
):
    """
    Tests that restarting a completed run does not re-train but still logs artifacts.
    """
    # --- First run (to completion) ---
    run_name = "completed_test_run"
    mock_train_config.TRAINING_STEPS = 5
    with patch(
        "src.training.manager.TrainingManager._get_run_name", return_value=run_name
    ):
        run_single_training(mock_train_config, dummy_data_path, is_sweep=False)

    # --- Second run (attempt to restart) ---
    setup_mocks_for_training["log"].reset_mock()
    setup_mocks_for_training["log_artifact"].reset_mock()

    with patch(
        "src.training.manager.TrainingManager._get_run_name", return_value=run_name
    ):
        with patch("src.training.manager.TrainingManager.train") as mock_train:
            run_single_training(mock_train_config, dummy_data_path, is_sweep=False)
            # The train method should not have been called because the run is completed
            mock_train.assert_not_called()

    # Artifacts should still be logged again
    setup_mocks_for_training["log_artifact"].assert_called()
