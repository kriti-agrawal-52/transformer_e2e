import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from functools import partial

# Add src to path to allow direct import
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from training.manager import run_hyperparameter_search


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


@pytest.mark.integration
def test_sweep_initialization_and_run(
    mock_train_config, dummy_data_path, setup_mocks_for_sweep
):
    """Tests that a hyperparameter sweep is initialized and the agent runs."""
    mock_train_config.HP_SEARCH_MAX_JOBS = 3

    run_hyperparameter_search(mock_train_config, dummy_data_path)

    setup_mocks_for_sweep["sweep"].assert_called_once()
    sweep_config = setup_mocks_for_sweep["sweep"].call_args[0][0]

    assert sweep_config["method"] == "grid"
    assert "learning_rate" in sweep_config["parameters"]
    assert "batch_size" in sweep_config["parameters"]

    # Check that wandb.agent was called to execute the sweep
    agent_args = setup_mocks_for_sweep["agent"].call_args
    assert agent_args[0][0] == "test_sweep_id"
    assert agent_args[1]["count"] == mock_train_config.HP_SEARCH_MAX_JOBS
    assert agent_args[1]["project"] == mock_train_config.WANDB_PROJECT


@pytest.mark.integration
def test_sweep_respects_max_jobs(
    mock_train_config, dummy_data_path, setup_mocks_for_sweep
):
    """Tests that the number of jobs run by the sweep respects the config limit."""
    mock_train_config.HP_SEARCH_MAX_JOBS = 2

    # To test the number of calls, we need to make the mocked agent
    # actually invoke the function it's given.
    def agent_side_effect(sweep_id, function, count, project):
        for i in range(count):
            # In a real sweep, wandb would populate its config.
            # We simulate that here before calling the function.
            with patch(
                "wandb.config",
                {"learning_rate": 0.01, "batch_size": 8, "context_window": 32},
            ):
                function()

    setup_mocks_for_sweep["agent"].side_effect = agent_side_effect

    run_hyperparameter_search(mock_train_config, dummy_data_path)

    # The mocked training function should have been called 'max_jobs' times
    assert (
        setup_mocks_for_sweep["run_training"].call_count
        == mock_train_config.HP_SEARCH_MAX_JOBS
    )


@pytest.mark.integration
def test_empty_sweep_config_raises_error(mock_train_config, dummy_data_path):
    """Tests that the sweep handles an empty parameter configuration gracefully."""
    mock_train_config.HP_SEARCH_LRS = []
    mock_train_config.HP_SEARCH_BATCH_SIZES = []
    mock_train_config.HP_SEARCH_CONTEXT_WINDOWS = []

    with pytest.raises(ValueError, match="No hyperparameter search parameters defined"):
        run_hyperparameter_search(mock_train_config, dummy_data_path)


@pytest.mark.integration
def test_sweep_agent_calls_training_with_correct_params(
    mock_train_config, dummy_data_path, setup_mocks_for_sweep
):
    """
    Tests that the function executed by the agent correctly calls run_single_training
    with parameters updated from the sweep config.
    """

    # Simulate the agent running just once
    def agent_side_effect(sweep_id, function, count, project):
        with patch(
            "wandb.config",
            {"learning_rate": 0.01, "batch_size": 8, "context_window": 32},
        ):
            function()

    setup_mocks_for_sweep["agent"].side_effect = agent_side_effect

    run_hyperparameter_search(mock_train_config, dummy_data_path)

    setup_mocks_for_sweep["run_training"].assert_called_once()
    call_args = setup_mocks_for_sweep["run_training"].call_args[0]

    called_config = call_args[0]

    # The config passed to the training function should be updated with sweep values
    assert called_config.LEARNING_RATE == 0.01
    assert called_config.BATCH_SIZE == 8
    assert called_config.CONTEXT_WINDOW == 32
    assert called_config.TRAINING_STEPS == mock_train_config.HP_SEARCH_STEPS
    assert call_args[2] is True  # is_sweep should be True
