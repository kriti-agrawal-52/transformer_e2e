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

### Hyperparameter Sweep

Run a hyperparameter search (sweep):

```sh
python -m transformer_e2e.scripts.train --mode sweep --config transformer_e2e/configs/config.yml
```

- Sweep settings are controlled in `config.yml` under the `HP_SEARCH_*` keys.

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

- Run all tests:
  ```sh
  pytest
  ```
- CI (GitHub Actions) automatically checks:
  - Code formatting (`black`)
  - Linting (`flake8`)
  - All tests (`pytest`)

## Notes

- All paths are relative to the project root.
- Checkpoints and logs are managed automatically.
- For advanced usage, see comments in the config files and scripts.
- For troubleshooting, check the log files specified in the configs.

---
