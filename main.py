import torch
import logging
import os
import argparse
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from model import TransformerModel
from data_utils import PreprocessingTraining
from train import train_model, hyperparameter_search, evaluate_validation_loss # Import train_model and hyperparameter_search
import config

# Setting up logger (centralized)
logging.basicConfig(
    filename='transformer_main.log', # Unified log file
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def run_training(args):
    # Use config values, but allow args to override
    wandb_project = args.wandb_project if args.wandb_project else config.WANDB_PROJECT
    wandb_entity = args.wandb_entity if args.wandb_entity else config.WANDB_ENTITY
    
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name="wikitext-2_main_training_run_with_dropout",
        notes="Training a decoder-only multi-head, multi-layer transformer on WikiText-2 raw using GPT2 tokenizer.",
        tags=["transformer", "wikitext2", "language_model", "pytorch", "main_run"],
        reinit=True
    )

    try:
        logger.info("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        raw_text = "\n".join(dataset['train']['text'])
        logger.info("Successfully loaded WikiText-2 dataset.")
        # we are reducing the size of the dataset since we are training on a cpu
        target_num_chars = config.MAX_CHARS_FOR_TRAINING
        if len(raw_text) > target_num_chars:
            raw_text = raw_text[:target_num_chars]
            logger.info(f"Limited raw text to first {target_num_chars} characters.")
        else:
            logger.info(f"Raw text is smaller than a million characters, we can use it as it is for training on a CPU.")
            
        wandb.config.update({"dataset_size_chars": len(raw_text), "dataset_name": "WikiText-2 Raw (Limited)"})
        
    except Exception as e:
        logger.critical(f"Could not load WikiText-2 dataset. Error: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Could not load WikiText-2 dataset. Ensure internet connection or datasets library is installed.")
        wandb.finish(exit_code=1)
        return

    #torch.manual_seed(1337)

    tokenizer_name = config.TOKENIZER_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
        wandb.config.update({"tokenizer_name": tokenizer_name, "tokenizer_vocab_size": tokenizer.vocab_size})
    except Exception as e:
        logger.critical(f"Could not load tokenizer '{tokenizer_name}'. Error: {e}", exc_info=True)
        print(f"Error: Could not load tokenizer '{tokenizer_name}'. Ensure internet access or cached tokenizer.")
        wandb.finish(exit_code=1)
        return
    
    # -- Load hyperparameters from args (which defaults to config.py values) ---
    batch_s = args.batch_size
    time_s = args.context_window
    channel_d = args.channel_dim
    num_h = args.num_heads
    num_l = args.num_layers
    learning_r = args.learning_rate
    training_steps = args.training_steps
    val_check = args.val_check_every
    train_patience = args.patience

    wandb.config.update({
        "batch_size": batch_s, "context_window": time_s, "embedding_dim": channel_d,
        "num_heads": num_h, "num_layers": num_l, "learning_rate": learning_r,
        "training_steps": training_steps, "validation_check_every": val_check,
        "early_stopping_patience": train_patience, "device": str(DEVICE)
    })
    
    prep = PreprocessingTraining(raw_text, tokenizer = tokenizer, batch_size=batch_s, time_steps=time_s)
    
    model = TransformerModel(
        vocab_size=prep.vocab_size, channel_dim=channel_d, context_window=prep.time_steps,
        num_heads=num_h, num_layers=num_l
    ).to(DEVICE)
    
    num_model_params = sum(p.numel() for p in model.parameters())/1e6
    logger.info(f"Model initialized with {num_model_params:.2f}M parameters on {DEVICE}.")
    wandb.config.update({"total_model_parameters_M": num_model_params})

    try:
        xb, yb = prep.get_batch("train")
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        expected_initial_loss = torch.log(torch.tensor(prep.vocab_size, dtype=torch.float)).item()
        _, loss_before = model(xb, yb)
        logger.info(f"Initial loss before training: {loss_before.item():.4f} (Expected for random ~{expected_initial_loss:.2f})")
        print(f"Initial loss before training: {loss_before.item():.4f} (Expected for random ~{expected_initial_loss:.2f})")
        wandb.log({"initial_loss": loss_before.item()})
    except Exception as e:
        logger.error(f"Could not perform initial loss check: {e}", exc_info=True)

    logger.info("Starting training...")
    train_model(model, prep, steps=training_steps, val_check_every=val_check, patience=train_patience, lr=learning_r)
    logger.info("Training finished.")

    # Load the best model saved during training for subsequent evaluations
    checkpoint_filename_main_artifact = f"transformer_bs{batch_s}_cw{time_s}_lr{learning_r:.0e}_best.pt"
    checkpoints_dir_main_artifact = "model_checkpoints" 
    checkpoint_path_for_artifact_main = os.path.join(checkpoints_dir_main_artifact, checkpoint_filename_main_artifact)

    try: 
        model.load_state_dict(torch.load(checkpoint_path_for_artifact_main, map_location=DEVICE))
        logger.info("Loaded best model from checkpoint for final evaluations.")
    except FileNotFoundError:
        logger.warning(f"Checkpoint file not found at {checkpoint_path_for_artifact_main}. Using current model state for evaluations.")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)

    if 'xb' in locals() and 'yb' in locals():
        _, loss_after = model(xb, yb)
        logger.info(f"Post-training loss (on one sample train batch): {loss_after.item():.4f}")
        print(f"Post-training loss (on one sample train batch): {loss_after.item():.4f}")
        wandb.log({"post_training_sample_train_loss": loss_after.item()})

    logger.info("Evaluating on test set...")
    test_loss_avg = evaluate_validation_loss(model, prep, split='test', eval_iters=50)
    if test_loss_avg is not None and not torch.isnan(torch.tensor(test_loss_avg)):
        formatted_test_loss = f"{test_loss_avg:.4f}"
        wandb.log({"test_loss_avg": test_loss_avg})
    else:
        formatted_test_loss = 'N/A'
        wandb.log({"test_loss_avg": float('nan')})

    logger.info(f"Average Test Set Loss: {formatted_test_loss}")
    print(f"Average Test Set Loss: {formatted_test_loss}")

    # Save Final Trained Model as a W&B Artifact
    model_artifact_main = wandb.Artifact(
        name=f"model-WikiText-2-bs{batch_s}-cw{time_s}-lr{learning_r:.0e}",
        type="model",
        description=f"Decoder-only Transformer trained on WikiText-2 Raw with HPs: BS={batch_s}, CW={time_s}, LR={learning_r}"
    )
    model_artifact_main.add_file(checkpoint_path_for_artifact_main)
    wandb.log_artifact(model_artifact_main)
    logger.info(f"Final model artifact logged for main run: {model_artifact_main.name}")
    wandb.finish()

def run_hyperparameter_search(args):
    # You'd typically re-init wandb here for each tuning run if hyperparameter_search isn't handling it
    # or ensure hyperparameter_search is set up to manage W&B runs for each combo
    
    # For simplicity, we'll assume hyperparameter_search manages its own wandb.init/finish calls per sub-run
    
    # Load dataset and tokenizer just once for the search
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        raw_text = "\n".join(dataset['train']['text'])
        logger.info("Successfully loaded WikiText-2 dataset for HP search.")
        
        # limit the size of dataset to 1 million characters
        target_num_chars = config.MAX_CHARS_FOR_TRAINING
        if len(raw_text) > target_num_chars:
            raw_text= raw_text[:target_num_chars]
            logger.info(f"Limited HP search raw text to first {target_num_chars} characters.")
        else:
            logger.info(f"HP search raw text is smaller than {target_num_chars} characters. Using all {len(raw_text)} characters.")
    except Exception as e:
        logger.critical(f"Could not load WikiText-2 dataset for HP search. Error: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Could not load WikiText-2 dataset for HP search.")
        return

    tokenizer_name = config.TOKENIZER_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token}) for HP search")
    except Exception as e:
        logger.critical(f"Could not load tokenizer '{tokenizer_name}' for HP search. Error: {e}", exc_info=True)
        print(f"Error: Could not load tokenizer '{tokenizer_name}' for HP search.")
        return

    logger.info("Starting hyperparameter search...")
    tuning_results = hyperparameter_search(
        raw_text,
        tokenizer,
        lrs=args.lrs,
        batch_sizes=args.batch_sizes,
        time_steps_list=args.context_windows
    )

    if tuning_results:
        valid_tuning_results = [r for r in tuning_results if r['val_loss'] is not None and not torch.isnan(torch.tensor(r['val_loss']))]
        if valid_tuning_results:
            sorted_results = sorted(valid_tuning_results, key=lambda x: x['val_loss'])
            print("\n--- Top Hyperparameter Configurations ---")
            for config in sorted_results[:3]:
                print(f"Val Loss: {config['val_loss']:.4f} | Batch: {config['batch_size']}, Context: {config['time_steps']}, LR: {config['learning_rate']}")
            logger.info("Hyperparameter search completed.")
        else:
            print("\nNo valid results from hyperparameter search to display.")
            logger.info("Hyperparameter search yielded no valid (non-NaN) results.")
    else:
        print("\nNo results from hyperparameter search.")
        logger.info("Hyperparameter search yielded no results.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Transformer Model Training or Hyperparameter Search")
    parser.add_argument('--mode', type=str, choices=['train', 'tune'], default='train',
                        help='Mode to run: "train" for main training, "tune" for hyperparameter search.')

    # # --- Training/Model HPs (default values from config.py) ---
    parser.add_argument('--wandb_project', type=str, default=config.WANDB_PROJECT,
                        help='WandB project name. Defaults to config.WANDB_PROJECT.')
    parser.add_argument('--wandb_entity', type=str, default=config.WANDB_ENTITY,
                        help='WandB entity (username/team). Defaults to config.WANDB_ENTITY.')
    parser.add_argument('--batch_size', type=int, default=config.TRAIN_BATCH_SIZE,
                        help='Batch size for training. Defaults to config.TRAIN_BATCH_SIZE.')
    parser.add_argument('--context_window', type=int, default=config.TRAIN_CONTEXT_WINDOW,
                        help='Context window (sequence length) for training. Defaults to config.TRAIN_CONTEXT_WINDOW.')
    parser.add_argument('--channel_dim', type=int, default=config.TRAIN_CHANNEL_DIM,
                        help='Embedding dimension. Defaults to config.TRAIN_CHANNEL_DIM.')
    parser.add_argument('--num_heads', type=int, default=config.TRAIN_NUM_HEADS,
                        help='Number of attention heads. Defaults to config.TRAIN_NUM_HEADS.')
    parser.add_argument('--num_layers', type=int, default=config.TRAIN_NUM_LAYERS,
                        help='Number of transformer layers. Defaults to config.TRAIN_NUM_LAYERS.')
    parser.add_argument('--learning_rate', type=float, default=config.TRAIN_LEARNING_RATE,
                        help='Learning rate for training. Defaults to config.TRAIN_LEARNING_RATE.')
    parser.add_argument('--training_steps', type=int, default=config.TRAIN_STEPS,
                        help='Total training steps. Defaults to config.TRAIN_STEPS.')
    parser.add_argument('--val_check_every', type=int, default=config.TRAIN_VALIDATION_CHECK_EVERY,
                        help='Validate every N steps. Defaults to config.TRAIN_VALIDATION_CHECK_EVERY.')
    parser.add_argument('--patience', type=int, default=config.TRAIN_EARLY_STOPPING_PATIENCE,
                        help='Early stopping patience. Defaults to config.TRAIN_EARLY_STOPPING_PATIENCE.')

    # --- Hyperparameter Search specific arguments (default values from config.py) ---
    parser.add_argument('--lrs', nargs='+', type=float, default=config.HP_SEARCH_LRS,
                        help='List of learning rates for hyperparameter search. Defaults to config.HP_SEARCH_LRS.')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=config.HP_SEARCH_BATCH_SIZES,
                        help='List of batch sizes for hyperparameter search. Defaults to config.HP_SEARCH_BATCH_SIZES.')
    parser.add_argument('--context_windows', nargs='+', type=int, default=config.HP_SEARCH_CONTEXT_WINDOWS,
                        help='List of context windows for hyperparameter search. Defaults to config.HP_SEARCH_CONTEXT_WINDOWS.')


    args = parser.parse_args()

    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'tune':
        run_hyperparameter_search(args)