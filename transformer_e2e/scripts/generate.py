import torch
import wandb
import argparse
import logging
import sys
from transformers import AutoTokenizer
import os
from src.models.transformer import TransformerModel
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# load configurations
generate_cfg = load_config("configs/generate_config.yml")


def setup_logging(cfg):
    logging.basicConfig(
        filename=cfg.LOG_FILE,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Add console handler to display log messages on console as well for debugging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)


def get_model_config_from_wandb(run_id, project_name):
    """
    Fetches the configuration dictionary from a specific W&B run."""
    try:
        logger.info(
            f"Attempting to fetch config for run_id: {run_id} and project: {project_name}"
        )
        runs = wandb.runs(project_name, {"_id": run_id})
        if not runs:
            logger.warning(
                f"Could not find run_id '{run_id}' in project '{project_name}'. Trying a broader search."
            )
            # broaden search to look for run_id in all projects
            runs = wandb(filters={"_id": run_id})

        if not runs:
            logger.error(
                f"Could not find any run with W&B run_id: {run_id}. "
                f"Ensure you are logged in and the project/run_id is correct."
            )
            raise ValueError

        if len(runs) > 1:
            logger.warning(
                f"Found multiple runs with ID {run_id}. Using the first one: {runs[0].path}"
            )

        run = runs[0]
        model_config = run.config
        logger.info(f"Successfully retrieved config from run: {run.name} ({run.id})")
        return model_config
    except wandb.errors.CommError as e:
        logger.error(f"W&B communication error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch config from W&B: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Transformer model via config file."
    )
    parser.add_argument(
        "--gen_config",
        type=str,
        default="generate_config.yml",
        help="Path to the generation configuration YAML file.",
    )
    args = parser.parse_args()

    # Load generation configuration
    try:
        logger.info(f"Loading generation configuration from: {args.gen_config}")
        generate_cfg = load_config(args.gen_config)
        setup_logging(generate_cfg)  # Set up logging after loading config
    except Exception as e:
        logger.error(f"Failed to load generation config: {e}")
        raise

    # Checking if checkpoint exists
    if not os.path.exists(generate_cfg.CHECKPOINT_PATH):
        logger.error(f"Checkpoint file not found: {generate_cfg.CHECKPOINT_PATH}")
        raise FileNotFoundError

    # Load local training config to get W&B project name
    local_cfg_dict = load_config(generate_cfg.MAIN_CONFIG_PATH)
    wandb_project = local_cfg_dict.get("WANDB_PROJECT")
    if not wandb_project:
        logger.error(
            f"WANDB_PROJECT not found in local config file: {generate_cfg.MAIN_CONFIG_PATH}"
        )
        sys.exit(1)

    # Get the model's training configuration from W&B
    model_config = get_model_config_from_wandb(generate_cfg.RUN_ID, wandb_project)
    tokenizer_name = model_config.get(
        "tokenizer_name", local_cfg_dict.get("TOKENIZER_NAME", "gpt2")
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Recreate model architecture using W&B config
    logger.info("Recreating model architecture...")
    model = TransformerModel(
        vocab_size=vocab_size,
        channel_dim=model_config["channel_dim"],
        context_window=model_config["context_window"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
    )

    # Load state dict from local checkpoint
    logger.info(
        f"Loading model state from local checkpoint: {generate_cfg.CHECKPOINT_PATH}"
    )
    checkpoint = torch.load(
        generate_cfg.CHECKPOINT_PATH, map_location=generate_cfg.DEVICE
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(generate_cfg.DEVICE)
    model.eval()
    logger.info("Model successfully loaded.")

    # Encode the prompt
    input_ids = torch.tensor([tokenizer.encode(generate_cfg.PROMPT)], dtype=torch.long)

    # Generate text
    logger.info("Generating text...")
    generated_ids = model.generate(
        input_ids,
        generate_cfg.MAX_TOKENS,
        generate_cfg.TEMPERATURE,
        generate_cfg.TOP_K,
        generate_cfg.DEVICE,
    )

    # Decode and print
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    print("\n--- Generated Text ---")
    print(generated_text)
    print("--- End ---")


if __name__ == "__main__":
    main()
