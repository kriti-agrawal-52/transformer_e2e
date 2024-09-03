# train.py
import argparse
import logging
import wandb
import sys

from src.utils.config_loader import load_config
from src.data.processing import setup_data_and_tokenizer
from src.training.manager import run_single_training, run_hyperparameter_search


def main():

    parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "sweep"],
        default="single",
        help="Training mode: 'single' for one single run, 'sweep' for hyperparameter search",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Setup logging
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

    logger = logging.getLogger(__name__)
    logger.info(f"Starting training process with mode: {args.mode}")

    raw_text, tokenizer = None, None

    try:
        # Load data based on config or mode
        should_run = (args.mode == "single" and cfg.SHOULD_TRAIN_SINGLE_RUN) or (
            args.mode == "sweep" and cfg.SHOULD_HYPERPARAMETER_SEARCH
        )
        if should_run:
            raw_text, tokenizer = setup_data_and_tokenizer(cfg)

        # execute based on mode and config
        if args.mode == "single":
            if cfg.SHOULD_TRAIN_SINGLE_RUN and raw_text:
                run_single_training(raw_text, tokenizer, cfg)
            else:
                logger.info(
                    "Single training is disabled in config or mode doesn't match. Exiting."
                )
        elif args.mode == "sweep":
            if cfg.SHOULD_HYPERPARAMETER_SEARCH and raw_text:
                run_hyperparameter_search(raw_text, tokenizer, cfg)
            else:
                logger.info(
                    "Hyperparameter tuning is disabled in config or mode doesn't match. Exiting."
                )
    except Exception as e:
        logger.critical(f"Unhandled error in single run: {e}", exc_info=True)
        print(f"An unhandled error occurred: {e}")
    finally:
        # Final cleanup for W&B if a run is still active
        if wandb.run:
            wandb.finish()
        logger.info("Training process finished.")


if __name__ == "__main__":
    main()
