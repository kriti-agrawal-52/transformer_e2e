import torch
import wandb
import argparse
import logging
import sys
from transformers import AutoTokenizer
import os
from src.models.transformer import TransformerModel
from src.utils.config_loader import load_config
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def validate_generation_params(cfg):
    """
    Validates generation parameters from config. Raises ValueError if invalid.
    """
    errors = []
    if not (0.0 <= cfg.TEMPERATURE <= 2.0):
        errors.append(f"TEMPERATURE ({cfg.TEMPERATURE}) must be between 0.0 and 2.0.")
    if cfg.TOP_K is not None and (not isinstance(cfg.TOP_K, int) or cfg.TOP_K < 0):
        errors.append(f"TOP_K ({cfg.TOP_K}) must be a non-negative integer or None.")
    if not (1 <= cfg.MAX_TOKENS <= 512):
        errors.append(f"MAX_TOKENS ({cfg.MAX_TOKENS}) must be between 1 and 512.")
    if not cfg.PROMPT or not isinstance(cfg.PROMPT, str):
        errors.append("PROMPT must be a non-empty string.")
    if errors:
        raise ValueError("Invalid generation parameters:\n" + "\n".join(errors))

# load configurations
# (moved into main for better error handling)
# generate_cfg = load_config("configs/generate_config.yml")

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
    Fetches model configuration from a W&B run, with robust error handling for
    connection, project, and run ID issues.
    """
    logger.info(f"Attempting to fetch config for run '{run_id}' from W&B project '{project_name}'...")
    try:
        # This is the first point of failure if the API key is wrong or there's no connection.
        api = wandb.Api()

        # This is the second point of failure if the project/run path is incorrect.
        run = api.run(f"{project_name}/{run_id}")

        # The api.run() call can return None if the run is not found.
        if run is None:
            logger.error(
                f"Run with ID '{run_id}' was not found in project '{project_name}'. "
                "Please verify the run ID and project name."
            )
            return None
        logger.info(f"Successfully fetched config for run '{run.name}' ({run.id}) from W&B.")
        return run.config  # Return as plain dict
    except wandb.errors.CommError as e:
        logger.error(
            "Could not connect to W&B API. This may be due to a missing or "
            "invalid API key, or a network issue. Please ensure your "
            "WANDB_API_KEY is set correctly in your .env file."
        )
        logger.error(f"Original W&B error: {e}")
        return None  # Return None to signal failure

    except Exception as e:
        # Catch any other unexpected errors during the process.
        logger.error(f"An unexpected error occurred while fetching config from W&B for run '{run_id}': {e}", exc_info=True)
        return None


def main():
    """
    Main function to generate text from a trained model.
    Improved for robust error handling, parameter validation, and resource cleanup.
    """
    # Load environment variables from .env file
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate text from a Transformer model.")
    parser.add_argument(
        "--gen_config",
        type=str,
        default="generate_config.yml",
        help="Path to the generation configuration YAML file.",
    )
    args = parser.parse_args()

    generate_cfg = None
    wandb_run = None
    try:
        # Load generation configuration
        logger.info(f"Loading generation configuration from: {args.gen_config}")
        generate_cfg = load_config(args.gen_config)
        setup_logging(generate_cfg)  # Set up logging after loading config

        # Validate generation parameters before proceeding
        validate_generation_params(generate_cfg)

        # Checking if checkpoint exists
        if not os.path.exists(generate_cfg.CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found: {generate_cfg.CHECKPOINT_PATH}")

        # Load local training config to get W&B project name
        local_cfg_dict = load_config(generate_cfg.MAIN_CONFIG_PATH)
        wandb_project = local_cfg_dict.get("WANDB_PROJECT")
        if not wandb_project:
            raise RuntimeError(f"WANDB_PROJECT not found in local config file: {generate_cfg.MAIN_CONFIG_PATH}")

        # Get model hyperparameters from the specified W&B run
        model_config = get_model_config_from_wandb(generate_cfg.RUN_ID, wandb_project)
        if model_config is None:
            raise RuntimeError("Could not retrieve model configuration from W&B. Exiting.")

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
        try:
            logger.info(
                f"Loading model state from local checkpoint: {generate_cfg.CHECKPOINT_PATH}"
            )
            checkpoint = torch.load(
                generate_cfg.CHECKPOINT_PATH, map_location=generate_cfg.DEVICE
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found at: {generate_cfg.CHECKPOINT_PATH}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model checkpoint: {e}", exc_info=True)
            raise

        model.to(generate_cfg.DEVICE)
        model.eval()
        logger.info("Model loaded and moved to device.")

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

    except Exception as e:
        logger.critical(f"Generation failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
    finally:
        # Ensure W&B run is closed if it was started (for future extensibility)
        if wandb.run:
            wandb.finish()
        logger.info("Generation process finished.")


if __name__ == "__main__":
    main()
