import pytest
import os
import yaml
from types import SimpleNamespace
from unittest.mock import patch, mock_open

from src.utils.config_loader import load_config


@pytest.mark.unit
def test_load_config_success(tmp_path):
    """
    Tests that load_config successfully loads a YAML file and returns a SimpleNamespace object.
    """
    config_path = tmp_path / "config.yml"
    config_content = {
        "key": "value", 
        "nested": {"n_key": "n_value"},
        "DEVICE": "cpu",
        "MODEL_CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
        "LOSS_PLOT_DIRECTORY": str(tmp_path / "plots")
    }

    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    config = load_config(str(config_path))

    assert isinstance(config, SimpleNamespace)
    assert config.key == "value"
    assert config.nested.n_key == "n_value"


@pytest.mark.unit
def test_load_config_file_not_found():
    """
    Tests that load_config raises a FileNotFoundError for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yml")


@pytest.mark.unit
def test_load_config_empty_file(tmp_path):
    """
    Tests how load_config handles an empty YAML file.
    It should handle None config gracefully.
    """
    config_path = tmp_path / "empty_config.yml"
    config_path.touch()

    # Empty YAML files load as None, which should be handled
    with pytest.raises((TypeError, AttributeError)):
        load_config(str(config_path))


@pytest.mark.unit
def test_load_config_yaml_error(tmp_path):
    """
    Tests that load_config raises a YAMLError for a malformed YAML file.
    """
    config_path = tmp_path / "bad_config.yml"
    with open(config_path, "w") as f:
        f.write("key: value: another_value")  # Malformed YAML

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_path))


@pytest.mark.unit  
def test_load_config_missing_required_keys(tmp_path):
    """
    Tests behavior when required config keys are missing.
    """
    config_path = tmp_path / "incomplete_config.yml"
    incomplete_config = {
        "DEVICE": "cpu",
        # Missing MODEL_CHECKPOINTS_DIR and LOSS_PLOT_DIRECTORY
    }
    
    with open(config_path, "w") as f:
        yaml.dump(incomplete_config, f)
    
    # Should raise KeyError when trying to access missing required paths
    with pytest.raises(KeyError):
        load_config(str(config_path))


@pytest.mark.unit
def test_load_config_invalid_wandb_project_name(tmp_path):
    """
    Tests that invalid W&B project names are rejected.
    """
    config_path = tmp_path / "invalid_wandb_config.yml"
    config_content = {
        "WANDB_PROJECT": "invalid project name with spaces!@#",
        "DEVICE": "cpu",
        "MODEL_CHECKPOINTS_DIR": "/tmp/checkpoints",
        "LOSS_PLOT_DIRECTORY": "/tmp/plots"
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    with pytest.raises(ValueError, match="Invalid WANDB_PROJECT name"):
        load_config(str(config_path))


@pytest.mark.unit
def test_load_config_auto_device_selection(tmp_path):
    """
    Tests that DEVICE='auto' is correctly resolved to cuda or cpu.
    """
    config_path = tmp_path / "auto_device_config.yml" 
    config_content = {
        "DEVICE": "auto",
        "WANDB_PROJECT": "test-project",
        "MODEL_CHECKPOINTS_DIR": "/tmp/checkpoints",
        "LOSS_PLOT_DIRECTORY": "/tmp/plots"
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    config = load_config(str(config_path))
    
    # Should be resolved to either "cuda" or "cpu"
    assert config.DEVICE in ["cuda", "cpu"]
