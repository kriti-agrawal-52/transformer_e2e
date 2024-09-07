import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import os
import sys

# Add scripts and src to path to allow direct import
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

# The script to be tested
from generate import validate_generation_params, get_model_config_from_wandb
import wandb  # Import wandb to mock its exceptions

# --- Fixtures ---


@pytest.fixture
def base_generate_config():
    """Provides a base valid generation configuration."""
    return SimpleNamespace(
        TEMPERATURE=0.7, TOP_K=50, MAX_TOKENS=100, PROMPT="Hello, world!"
    )


# --- Tests for validate_generation_params ---


@pytest.mark.unit
def test_validate_generation_params_valid(base_generate_config):
    """Tests that no exception is raised for a valid configuration."""
    try:
        validate_generation_params(base_generate_config)
    except ValueError:
        pytest.fail("validate_generation_params raised ValueError unexpectedly!")


@pytest.mark.unit
@pytest.mark.parametrize("temp", [-0.1, 2.1])
def test_validate_generation_params_invalid_temperature(base_generate_config, temp):
    """Tests for invalid TEMPERATURE values."""
    base_generate_config.TEMPERATURE = temp
    with pytest.raises(ValueError, match="TEMPERATURE"):
        validate_generation_params(base_generate_config)


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [-1, 1.5, "abc"])
def test_validate_generation_params_invalid_top_k(base_generate_config, top_k):
    """Tests for invalid TOP_K values."""
    base_generate_config.TOP_K = top_k
    with pytest.raises(ValueError, match="TOP_K"):
        validate_generation_params(base_generate_config)


@pytest.mark.unit
@pytest.mark.parametrize("max_tokens", [0, 513])
def test_validate_generation_params_invalid_max_tokens(
    base_generate_config, max_tokens
):
    """Tests for invalid MAX_TOKENS values."""
    base_generate_config.MAX_TOKENS = max_tokens
    with pytest.raises(ValueError, match="MAX_TOKENS"):
        validate_generation_params(base_generate_config)


@pytest.mark.unit
@pytest.mark.parametrize("prompt", ["", None, 123])
def test_validate_generation_params_invalid_prompt(base_generate_config, prompt):
    """Tests for invalid PROMPT values."""
    base_generate_config.PROMPT = prompt
    with pytest.raises(ValueError, match="PROMPT"):
        validate_generation_params(base_generate_config)


@pytest.mark.unit
def test_validate_generation_params_multiple_errors(base_generate_config):
    """Tests that multiple validation errors are reported."""
    base_generate_config.TEMPERATURE = -1.0
    base_generate_config.PROMPT = ""
    with pytest.raises(ValueError) as excinfo:
        validate_generation_params(base_generate_config)
    assert "TEMPERATURE" in str(excinfo.value)
    assert "PROMPT" in str(excinfo.value)


# --- Tests for get_model_config_from_wandb ---


@pytest.mark.unit
def test_get_model_config_from_wandb_success():
    """Tests successful retrieval of config from W&B."""
    with patch("wandb.Api") as mock_wandb_api:
        mock_run = MagicMock()
        mock_run.config = {"lr": 0.01, "epochs": 5}
        mock_run.name = "test-run"
        mock_run.id = "run123"

        mock_api_instance = mock_wandb_api.return_value
        mock_api_instance.run.return_value = mock_run

        config = get_model_config_from_wandb("run123", "test-project")

        mock_api_instance.run.assert_called_with("test-project/run123")
        assert config == {"lr": 0.01, "epochs": 5}


@pytest.mark.unit
def test_get_model_config_from_wandb_run_not_found():
    """Tests the case where the W&B run is not found."""
    with patch("wandb.Api") as mock_wandb_api:
        mock_api_instance = mock_wandb_api.return_value
        mock_api_instance.run.return_value = None  # Simulate run not found

        config = get_model_config_from_wandb("nonexistent-run", "test-project")

        assert config is None


@pytest.mark.unit
def test_get_model_config_from_wandb_comm_error():
    """Tests handling of a W&B communication error."""
    with patch(
        "wandb.Api", side_effect=wandb.errors.CommError("Connection failed")
    ) as mock_wandb_api:
        config = get_model_config_from_wandb("run123", "test-project")

        assert config is None


@pytest.mark.unit
def test_get_model_config_from_wandb_generic_exception():
    """Tests handling of an unexpected exception."""
    with patch(
        "wandb.Api", side_effect=Exception("Something went wrong")
    ) as mock_wandb_api:
        config = get_model_config_from_wandb("run123", "test-project")

        assert config is None
