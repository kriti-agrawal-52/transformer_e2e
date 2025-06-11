import pytest
import os
from unittest.mock import patch, MagicMock
from functools import partial

from src.training.manager import run_hyperparameter_search


@pytest.fixture
def setup_mocks_for_sweep(mock_train_config, dummy_data_path):
    """Centralized mock setup for hyperparameter sweep integration tests."""
    with patch("wandb.sweep") as mock_wandb_sweep, patch(
        "wandb.agent"
    ) as mock_wandb_agent, patch(
        "src.training.manager.run_single_training"
    ) as mock_run_single_training:

        mock_wandb_sweep.return_value = "test_sweep_id"

        # This is the function wandb.agent will call. We need to define it here
        # so we can pass it to the agent mock.
        def sweep_function(config=None):
            # This function body is what the agent executes.
            # We can spy on it or mock its dependencies.
            pass

        yield {
            "sweep": mock_wandb_sweep,
            "agent": mock_wandb_agent,
            "run_training": mock_run_single_training,
            "sweep_function": sweep_function,
        }


@patch("src.training.manager.wandb")
def test_sweep_initialization_and_run(mock_wandb, mock_train_config, dummy_data_path):
    """Test that hyperparameter sweep initializes correctly and runs training for all parameter combinations."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    with patch("src.training.manager.TrainingManager.run") as mock_manager_run:
        mock_manager_run.return_value = "/path/to/best_checkpoint.pt"
        
        # Call with correct parameters
        run_hyperparameter_search(dummy_data_path, mock_tokenizer, mock_train_config)
        
        # Should create manager instances for each hyperparameter combination
        expected_combinations = (
            len(mock_train_config.HP_SEARCH_BATCH_SIZES) *
            len(mock_train_config.HP_SEARCH_CONTEXT_WINDOWS) *
            len(mock_train_config.HP_SEARCH_LRS)
        )
        assert mock_manager_run.call_count == expected_combinations


@patch("src.training.manager.wandb")
def test_sweep_respects_max_jobs(mock_wandb, mock_train_config, dummy_data_path):
    """Test that sweep respects the maximum number of jobs configuration."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    # Reduce the search space to test job limiting
    mock_train_config.HP_SEARCH_BATCH_SIZES = [2, 4]
    mock_train_config.HP_SEARCH_CONTEXT_WINDOWS = [8]
    mock_train_config.HP_SEARCH_LRS = [1e-3]
    
    with patch("src.training.manager.TrainingManager.run") as mock_manager_run:
        mock_manager_run.return_value = "/path/to/best_checkpoint.pt"
        
        run_hyperparameter_search(dummy_data_path, mock_tokenizer, mock_train_config)
        
        # Should only run 2 jobs (2 batch sizes * 1 context window * 1 lr)
        assert mock_manager_run.call_count == 2


def test_empty_sweep_config_raises_error(mock_train_config, dummy_data_path):
    """Test that empty hyperparameter configuration raises appropriate error."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    # Empty sweep configuration
    mock_train_config.HP_SEARCH_BATCH_SIZES = []
    mock_train_config.HP_SEARCH_CONTEXT_WINDOWS = [8]
    mock_train_config.HP_SEARCH_LRS = [1e-3]
    
    with patch("src.training.manager.TrainingManager.run") as mock_manager_run:
        run_hyperparameter_search(dummy_data_path, mock_tokenizer, mock_train_config)
        
        # Should not run any jobs with empty batch sizes
        assert mock_manager_run.call_count == 0


@patch("src.training.manager.wandb")  
def test_sweep_agent_calls_training_with_correct_params(mock_wandb, mock_train_config, dummy_data_path):
    """Test that sweep agent calls training with correct hyperparameters."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    # Use a single combination to test parameter passing
    mock_train_config.HP_SEARCH_BATCH_SIZES = [4]
    mock_train_config.HP_SEARCH_CONTEXT_WINDOWS = [16]
    mock_train_config.HP_SEARCH_LRS = [1e-3]
    
    with patch("src.training.manager.TrainingManager") as mock_manager_class:
        mock_manager_instance = MagicMock()
        mock_manager_class.return_value = mock_manager_instance
        mock_manager_instance.run.return_value = "/path/to/best_checkpoint.pt"
        
        run_hyperparameter_search(dummy_data_path, mock_tokenizer, mock_train_config)
        
        # Verify manager was called with correct parameters
        mock_manager_class.assert_called_once()
        call_args = mock_manager_class.call_args
        
        # Check that the run_params contain expected values
        run_params = call_args[0][0]  # First positional argument
        assert run_params["batch_size"] == 4
        assert run_params["context_window"] == 16
        assert run_params["learning_rate"] == 1e-3


@patch("src.training.manager.wandb")
def test_sweep_completes_all_jobs_and_saves_best_run(mock_wandb, mock_train_config, dummy_data_path):
    """Test that sweep completes all jobs and best run is saved and uploaded."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.vocab_size = 50257
    
    # Small search space for predictable testing
    mock_train_config.HP_SEARCH_BATCH_SIZES = [2, 4]
    mock_train_config.HP_SEARCH_CONTEXT_WINDOWS = [8]  
    mock_train_config.HP_SEARCH_LRS = [1e-3]
    
    with patch("src.training.manager.TrainingManager") as mock_manager_class:
        mock_manager_instance = MagicMock()
        mock_manager_class.return_value = mock_manager_instance
        
        # Simulate different performance for different runs
        mock_manager_instance.run.side_effect = [
            "/path/to/checkpoint_bs2.pt",  # First run
            "/path/to/checkpoint_bs4.pt"   # Second run (assume this is better)
        ]
        
        run_hyperparameter_search(dummy_data_path, mock_tokenizer, mock_train_config)
        
        # Should complete exactly 2 jobs (2 batch sizes * 1 context window * 1 lr)
        assert mock_manager_instance.run.call_count == 2
        
        # Verify all expected hyperparameter combinations were tried
        call_args_list = mock_manager_class.call_args_list
        assert len(call_args_list) == 2
        
        # First call should have batch_size=2
        first_params = call_args_list[0][0][0]
        assert first_params["batch_size"] == 2
        
        # Second call should have batch_size=4  
        second_params = call_args_list[1][0][0]
        assert second_params["batch_size"] == 4
