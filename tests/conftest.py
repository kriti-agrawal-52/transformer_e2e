import pytest
from types import SimpleNamespace
import os
import shutil
import yaml


@pytest.fixture(scope="session")
def test_artifacts_dir(tmpdir_factory):
    """Create a temporary directory for test artifacts."""
    temp_dir = tmpdir_factory.mktemp("test_artifacts")
    yield str(temp_dir)
    shutil.rmtree(str(temp_dir), ignore_errors=True)


@pytest.fixture(scope="session")
def dummy_data_path(test_artifacts_dir):
    """Create a dummy data file for testing."""
    data_content = "This is a simple text for testing purposes. " * 100
    data_path = os.path.join(test_artifacts_dir, "dummy_data.txt")
    with open(data_path, "w") as f:
        f.write(data_content)
    return data_path


@pytest.fixture
def mock_config():
    """Returns a mutable mock config object as a SimpleNamespace."""
    config_dict = {
        "DEVICE": "cpu",
        "BATCH_SIZE": 4,
        "CONTEXT_WINDOW": 16,
        "CHANNEL_DIM": 32,
        "NUM_HEADS": 4,
        "NUM_LAYERS": 2,
        "LEARNING_RATE": 1e-3,
        "TRAINING_STEPS": 10,
        "VALIDATION_CHECK_EVERY": 5,
        "EARLY_STOPPING_PATIENCE": 3,
        "MIN_DELTA": 0.01,
        "EVAL_ITERS_VAL": 2,
        "EVAL_ITERS_TEST": 2,
        "MIN_SUCCESSFUL_VAL_BATCH_RATIO": 0.5,
        "DROPOUT_RATE": 0.1,
        "WANDB_PROJECT": "test-project",
        "WANDB_RUN_PREFIX": ["test-run", "test-tune"],
        "MODEL_CHECKPOINTS_DIR": "checkpoints",
        "LOSS_PLOT_DIRECTORY": "plots",
        "LOG_FILE": "test.log",
        "DELETE_LATEST_CHECKPOINT_ON_COMPLETION": True,
        "TOKENIZER_NAME": "gpt2",
        "RAW_TEXT_LIMIT": 1000,
        "HP_SEARCH_MAX_JOBS": 2,
        "HP_SEARCH_LRS": [1e-4, 1e-3],
        "HP_SEARCH_BATCH_SIZES": [2, 4],
        "HP_SEARCH_CONTEXT_WINDOWS": [8, 16],
        "HP_SEARCH_STEPS": 10,
        "HP_VALIDATION_CHECK_EVERY": 5,
        "HP_EARLY_STOPPING_PATIENCE": 3,
        "DATASET_NAME": "test-dataset",
    }
    return SimpleNamespace(**config_dict)


@pytest.fixture
def mock_train_config(mock_config, test_artifacts_dir):
    """Provides a mock training config for integration tests."""
    cfg = mock_config
    cfg.MODEL_CHECKPOINTS_DIR = os.path.join(test_artifacts_dir, "checkpoints")
    cfg.LOSS_PLOT_DIRECTORY = os.path.join(test_artifacts_dir, "plots")
    cfg.LOG_FILE = os.path.join(test_artifacts_dir, "train.log")
    os.makedirs(cfg.MODEL_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(cfg.LOSS_PLOT_DIRECTORY, exist_ok=True)
    return cfg


@pytest.fixture
def mock_generate_config(test_artifacts_dir):
    """Provides a mock generation config."""
    config_dict = {
        "CHECKPOINT_PATH": os.path.join(
            test_artifacts_dir, "checkpoints", "test_run_best.pt"
        ),
        "MAIN_CONFIG_PATH": "path/to/mock/main_config.yml",
        "RUN_ID": "test-run-id",
        "PROMPT": "Hello world",
        "MAX_TOKENS": 10,
        "TEMPERATURE": 0.7,
        "TOP_K": 50,
        "DEVICE": "cpu",
        "LOG_FILE": os.path.join(test_artifacts_dir, "generate.log"),
    }
    # Create a dummy main config file for generate script to load
    main_config_path = os.path.join(test_artifacts_dir, "main_config.yml")
    with open(main_config_path, "w") as f:
        yaml.dump({"WANDB_PROJECT": "test-project"}, f)

    config_dict["MAIN_CONFIG_PATH"] = main_config_path

    return SimpleNamespace(**config_dict)
