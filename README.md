# Custom Transformer MLOps Pipeline

## Overview

This project provides an end-to-end pipeline for building, training, and deploying a custom Transformer language model from scratch using the WikiText-2 dataset. It covers:

- Model training (single run & hyperparameter sweep)
- Checkpointing and recovery
- Text generation from trained models
- Experiment tracking with Weights & Biases (W&B)
- Configurable, reproducible MLOps workflows
- CI-ready, comprehensive test suite

## Project Structure

- `transformer_e2e/scripts/`: Main scripts for training and generation
- `transformer_e2e/configs/`: YAML config files for training and generation
- `transformer_e2e/model_checkpoints/`: Model checkpoints (created automatically)
- `tests/`: Pytest-based test suite

## Setup

1. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
2. **Set up environment variables**
   - Create a `.env` file with your W&B API key and any other secrets.

## Training the Model

### Single Run Training

Run a single training session with the default config:

```sh
python -m transformer_e2e.scripts.train --mode single --config transformer_e2e/configs/config.yml
```

- The config file controls all hyperparameters and run metadata.
- **To start a fresh run:**
  - Edit `RUN_NAME` or the relevant entry in `WANDB_RUN_PREFIX` in `config.yml` to a new, unique value.
  - This avoids overwriting previous runs/checkpoints.
- **Checkpoints** are saved in `transformer_e2e/model_checkpoints/` (created automatically if missing).

### Resuming Training

- If training is interrupted, simply re-run the same command:
  ```sh
  python -m transformer_e2e.scripts.train --mode single --config transformer_e2e/configs/config.yml
  ```
- The script will automatically detect and resume from the latest checkpoint (no extra arguments needed).
- **Completion Tracking**: The system tracks completed training runs independently of checkpoint files, so it will correctly skip re-training runs that have already completed (either by finishing all steps or early stopping).

### How Completion Tracking Works

The system uses persistent metadata files to track training completion:

1. **During Training**: When training completes successfully, metadata is saved to `model_checkpoints/completion_tracking/<run_id>_completed.json`
2. **After Upload**: Local checkpoint files are deleted after successful W&B artifact upload
3. **On Resume**: The system checks completion metadata BEFORE attempting to load checkpoints
4. **Automatic Skip**: If a run was already completed, training is skipped entirely

This prevents the circular dependency where completion status was stored in checkpoint files that got deleted upon completion.

### Hyperparameter Sweep

Run a hyperparameter search (sweep):

```sh
python -m transformer_e2e.scripts.train --mode sweep --config transformer_e2e/configs/config.yml
```

- Sweep settings are controlled in `config.yml` under the `HP_SEARCH_*` keys.

## Managing Training Runs

### Viewing Completed Runs

List all completed training runs:

```sh
python scripts/manage_completed_runs.py list
```

Check the status of a specific run:

```sh
python scripts/manage_completed_runs.py status "single_bs4_cw16_lr1e-03"
```

### Forcing Re-training

If you need to re-run a completed training session, clear its completion status:

```sh
python scripts/manage_completed_runs.py clear "single_bs4_cw16_lr1e-03"
```

After clearing, you can run the training command again and it will start fresh.

### Understanding Run IDs

Run IDs are automatically generated based on training parameters:

- **Single runs**: `single_bs{batch_size}_cw{context_window}_lr{learning_rate}`
- **Sweep runs**: `tune_{batch_size}_{context_window}_{learning_rate}`

Examples:

- `single_bs4_cw16_lr1e-03` (single run with batch_size=4, context_window=16, lr=0.001)
- `tune_8_32_5e-04` (sweep run with batch_size=8, context_window=32, lr=0.0005)

## Text Generation

Generate text from a trained model:

```sh
python -m transformer_e2e.scripts.generate --gen_config transformer_e2e/configs/generate_config.yml
```

- Edit `generate_config.yml` to set the checkpoint path, prompt, and generation parameters.
- The script fetches model hyperparameters from W&B using the `RUN_ID` and loads the local checkpoint.

## Configuration

- **Training config:** `transformer_e2e/configs/config.yml`
- **Generation config:** `transformer_e2e/configs/generate_config.yml`
- For a new run, change the run name or prefix in the config to avoid overwriting previous results.

## Testing & CI

Run all tests:

```sh
pytest
```

Run specific test categories:

```sh
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Smoke tests only
pytest tests/smoke/
```

CI (GitHub Actions) automatically checks:

- Code formatting (`black`)
- Linting (`flake8`)
- All tests (`pytest`)

## File Structure & Data Management

### Key Directories

- `model_checkpoints/`: Contains training checkpoints and completion tracking data
  - `completion_tracking/`: Persistent metadata about completed runs (JSON files)
  - `*.pt`: PyTorch checkpoint files (deleted after successful W&B artifact upload)
- `training_plots/`: Loss plots and training visualizations
- `wandb/`: Local W&B run data and logs

### Checkpoint Cleanup

- **Local checkpoints** are automatically deleted after successful upload to W&B artifacts
- **Completion metadata** persists in `completion_tracking/` to prevent accidental re-training
- This saves local storage while maintaining run history and preventing duplicate training

## Troubleshooting

### Common Issues

**"Run was previously completed" message:**

- This means the training run finished successfully and was marked as complete
- Use `python scripts/manage_completed_runs.py status <run_id>` to check details
- Use `python scripts/manage_completed_runs.py clear <run_id>` to force re-training

**Training doesn't resume properly:**

- Check if completion status exists: `python scripts/manage_completed_runs.py status <run_id>`
- Verify checkpoint files exist in `model_checkpoints/`
- Check logs for specific error messages

**W&B connection issues:**

- Ensure `.env` file contains valid `WANDB_API_KEY`
- Check network connectivity to wandb.ai
- Local checkpoints will NOT be deleted if W&B upload fails

### Debug Commands

```sh
# List all completed runs
python scripts/manage_completed_runs.py list

# Check specific run status
python scripts/manage_completed_runs.py status "your_run_id"

# Clear completion status to force re-training
python scripts/manage_completed_runs.py clear "your_run_id"

# View current config
cat transformer_e2e/configs/config.yml
```

## Notes

- All paths are relative to the project root
- Checkpoints and logs are managed automatically
- For advanced usage, see comments in the config files and scripts
- Check log files specified in configs for detailed troubleshooting

---
