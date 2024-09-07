import pytest
import os
import subprocess
import sys
import yaml
from types import SimpleNamespace

# Add the root of the project to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


@pytest.fixture(scope="module")
def dummy_configs(tmpdir_factory):
    """Create dummy config files for dry-run tests."""
    config_dir = tmpdir_factory.mktemp("configs")

    train_config = {
        "SHOULD_TRAIN_SINGLE_RUN": True,
        "SHOULD_HYPERPARAMETER_SEARCH": False,
        "DEVICE": "cpu",
        "DATASET_NAME": "dummy",
        "RAW_TEXT_FILE": "dummy_data.txt",
        "TOKENIZER_NAME": "gpt2",
        "BATCH_SIZE": 1,
        "CONTEXT_WINDOW": 8,
        "TRAINING_STEPS": 1,
        "VALIDATION_CHECK_EVERY": 1,
        "WANDB_PROJECT": "test-project-smoke",
        "MODEL_CHECKPOINTS_DIR": str(config_dir.join("checkpoints")),
        "LOG_FILE": str(config_dir.join("smoke.log")),
        "RUN_MODE": "single",
    }

    gen_config = {
        "CHECKPOINT_PATH": "dummy_checkpoint.pt",
        "MAIN_CONFIG_PATH": "dummy_main_config.yml",
        "RUN_ID": "smoke-test-run",
        "PROMPT": "test",
        "MAX_TOKENS": 2,
        "DEVICE": "cpu",
        "TEMPERATURE": 0.7,
        "TOP_K": 10,
        "LOG_FILE": str(config_dir.join("smoke_generate.log")),
    }

    train_config_path = config_dir.join("train_config.yml")
    gen_config_path = config_dir.join("gen_config.yml")

    with open(train_config_path, "w") as f:
        yaml.dump(train_config, f)

    with open(gen_config_path, "w") as f:
        yaml.dump(gen_config, f)

    # Create a dummy checkpoint and main config for the generation script
    os.makedirs(config_dir.join("checkpoints"), exist_ok=True)
    open(config_dir.join("dummy_checkpoint.pt"), "a").close()
    with open(config_dir.join("dummy_main_config.yml"), "w") as f:
        yaml.dump({"WANDB_PROJECT": "test-project-smoke"}, f)

    gen_config["CHECKPOINT_PATH"] = str(config_dir.join("dummy_checkpoint.pt"))
    gen_config["MAIN_CONFIG_PATH"] = str(config_dir.join("dummy_main_config.yml"))
    with open(gen_config_path, "w") as f:
        yaml.dump(gen_config, f)

    return SimpleNamespace(
        **{"train": str(train_config_path), "generate": str(gen_config_path)}
    )


@pytest.mark.smoke
def test_train_script_help():
    """Test that train.py runs with --help."""
    result = subprocess.run(
        ["python", "transformer_e2e/scripts/train.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage: train.py" in result.stdout


@pytest.mark.smoke
def test_generate_script_help():
    """Test that generate.py runs with --help."""
    result = subprocess.run(
        ["python", "transformer_e2e/scripts/generate.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage: generate.py" in result.stdout


@pytest.mark.smoke
def test_train_script_dry_run_config(dummy_configs):
    """Test train.py with a config file to ensure it starts without error."""
    # This is a dry run, so we expect it to fail finding data, but not parsing config.
    # We mock the actual training call away in integration tests.
    result = subprocess.run(
        ["python", "transformer_e2e/scripts/train.py", "--config", dummy_configs.train],
        capture_output=True,
        text=True,
        check=False,
    )
    # It should fail, but not with a config parsing error. A FileNotFoundError for the data is expected.
    assert (
        "FileNotFoundError" in result.stderr
        or "ValueError: Input text is empty" in result.stderr
    )
    assert result.returncode != 0


@pytest.mark.smoke
def test_generate_script_dry_run_config(dummy_configs):
    """Test generate.py with a config file to ensure it starts without error."""
    # We expect this to fail because W&B is not reachable without a key,
    # but it should get past config parsing and validation.
    result = subprocess.run(
        [
            "python",
            "transformer_e2e/scripts/generate.py",
            "--gen_config",
            dummy_configs.generate,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "Could not connect to W&B API" in result.stderr
    assert result.returncode != 0
