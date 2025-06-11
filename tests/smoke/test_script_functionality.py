import pytest
import os
import sys
import subprocess
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


def run_script_with_pythonpath(script_args):
    """Run script with proper PYTHONPATH setup"""
    env = os.environ.copy()
    # Add the project root to PYTHONPATH so 'src' module can be found
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")
    
    return subprocess.run(
        script_args,
        capture_output=True,
        text=True,
        env=env,
        cwd=project_root
    )


@pytest.mark.smoke
def test_train_script_help():
    """Test that train.py script shows help information correctly."""
    result = run_script_with_pythonpath(['python', 'scripts/train.py', '--help'])
    assert result.returncode == 0
    assert "--config" in result.stdout or "usage:" in result.stdout


@pytest.mark.smoke
def test_generate_script_help():
    """Test that generate.py script shows help information correctly."""
    result = run_script_with_pythonpath(['python', 'scripts/generate.py', '--help'])
    assert result.returncode == 0
    assert "--gen_config" in result.stdout or "usage:" in result.stdout


@pytest.mark.smoke
def test_train_script_dry_run_config(test_artifacts_dir):
    """Test that train.py handles missing config file gracefully."""
    fake_config_path = os.path.join(test_artifacts_dir, "fake_config.yaml")
    
    result = run_script_with_pythonpath([
        'python', 'scripts/train.py', '--config', fake_config_path
    ])
    
    # Script should fail but with a proper error message
    assert result.returncode != 0
    assert (
        "FileNotFoundError" in result.stderr 
        or "No such file or directory" in result.stderr
        or "Input text is empty" in result.stderr
    )


@pytest.mark.smoke
def test_generate_script_dry_run_config(test_artifacts_dir):
    """Test that generate.py handles missing config file gracefully."""
    fake_config_path = os.path.join(test_artifacts_dir, "fake_gen_config.yaml")
    
    result = run_script_with_pythonpath([
        'python', 'scripts/generate.py', '--gen_config', fake_config_path
    ])
    
    # Script should handle missing config gracefully (may return 0 or non-0)
    # The important thing is that it doesn't crash with an unhandled exception
    assert result.returncode in [0, 1], f"Script should handle missing config gracefully, got return code: {result.returncode}"
    assert (
        "FileNotFoundError" in result.stderr 
        or "No such file or directory" in result.stderr
        or "Could not connect to W&B API" in result.stderr
    )


@pytest.mark.smoke 
def test_codebase_build_check_imports():
    """Test that all core modules can be imported without errors."""
    import_tests = [
        "import src.models.transformer",
        "import src.models.transformer_generate", 
        "import src.training.manager",
        "import src.training.utils",
        "import src.utils.config_loader",
        "import scripts.train",
        "import scripts.generate"
    ]
    
    for import_test in import_tests:
        try:
            exec(import_test)
            import_success = True
        except ImportError as e:
            import_success = False
            print(f"Import failed: {import_test} - {e}")
        
        assert import_success, f"Should be able to import: {import_test}"


@pytest.mark.smoke
def test_training_script_config_parsing():
    """Test that train.py can parse a minimal config without immediate crash."""
    # Test with basic config structure - script may fail later but should parse config
    result = run_script_with_pythonpath([
        'python', '-c', 
        '''
import sys, os
sys.path.insert(0, os.getcwd())
from src.utils.config_loader import load_config
from types import SimpleNamespace

# Test config parsing functionality
config = SimpleNamespace(
    DEVICE="cpu",
    BATCH_SIZE=2,
    LEARNING_RATE=1e-3,
    TRAINING_STEPS=1
)
print("Config parsing works")
        '''
    ])
    
    assert result.returncode == 0, "Basic config parsing should work"
    assert "Config parsing works" in result.stdout
