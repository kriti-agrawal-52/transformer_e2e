import pytest
import os
import yaml
from box import Box
from unittest.mock import patch, mock_open

from src.utils.config_loader import load_config


@pytest.mark.unit
def test_load_config_success(tmp_path):
    """
    Tests that load_config successfully loads a YAML file and returns a Box object.
    """
    config_path = tmp_path / "config.yml"
    config_content = {"key": "value", "nested": {"n_key": "n_value"}}

    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    config = load_config(str(config_path))

    assert isinstance(config, Box)
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
    It should return an empty Box object.
    """
    config_path = tmp_path / "empty_config.yml"
    config_path.touch()

    config = load_config(str(config_path))

    assert isinstance(config, Box)
    assert not config


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
