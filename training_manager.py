# training_manager.py
import torch
import torch.optim as optim
import wandb
import os
import logging

from model import TransformerModel
from utils_data import PreprocessingTraining
from utils_train import train_loop, load_checkpoint, evaluate_validation_loss

logger = logging.getLogger(__name__)

def run_post_training_eval(model, prep, best_model_path, cfg, device):
        """Loads the best model and runs test evaluation and text generation."""
        logger.info("Running post-training evaluations...")
        try:
            # Load the best model state
            best_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from {best_model_path} for final eval.")

            # Evaluate on test set
            test_loss = evaluate_validation_loss(model, prep, device, "test", cfg.EVAL_ITERS_TEST)
            test_loss_str = (
                f"{test_loss:.4f}"
                if not torch.isnan(torch.tensor(test_loss))
                else "N/A"
            )
            logger.info(f"Average Test Set Loss: {test_loss_str}")
            wandb.log({"test_loss_avg": test_loss})

        except Exception as e:
            logger.error(f"Post-training evaluation failed: {e}", exc_info=True)
            
def log_artifact(checkpoint_path, run_id, run_params, cfg):
    """Logs the best model checkpoint as a W&B Artifact."""
    if not os.path.exists(checkpoint_path):
        logger.warning(
            f"Checkpoint {checkpoint_path} not found for artifact logging."
        )
        return

    logger.info(f"Logging model artifact from {checkpoint_path}")
    try:
        artifact_name = f"{cfg.WANDB_PROJECT}_model-{run_id}" 
        # Using wandb.Artifact explicitly registers the model in the W&B Model Registry.
        # This is done once after the single training run completes.
        model_artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Transformer model from run {run_id}",
            metadata=run_params,  # log the params used for this model
        )
        model_artifact.add_file(checkpoint_path)
        wandb.log_artifact(model_artifact, aliases = ['best', run_id])
        logger.info(f"Logged artifact: {artifact_name}")
    except Exception as e:
        logger.error(f"Failed to log artifact: {e}", exc_info=True)


class TrainingManager:
    """Manages the complete training session end to end. """
    def __init__(self, run_params, tokenizer, raw_text, cfg, is_single_run=True):
        self.params = run_params
        self.tokenizer = tokenizer
        self.raw_text = raw_text
        self.cfg = cfg
        self.is_single_run = is_single_run
        self.run_id = self.params["run_id"]
        self.device = torch.device(cfg.DEVICE)
        
        bs = self.params["batch_size"]
        cw = self.params["context_window"]
        lr = self.params["learning_rate"]
        base_filename = f"run_{self.run_id}_bs{bs}_cw{cw}_lr{lr:.0e}"
        self.latest_ckpt_path = os.path.join(self.cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_latest.pt")
        self.best_ckpt_path = os.path.join(self.cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_best.pt")
        
    def run(self):
        """Executes the entire training session."""
        wandb.init(
            project=self.cfg.WANDB_PROJECT,
            name=self.params["name"],
            notes=self.params.get("notes", ""),
            tags=self.params.get("tags", []),
            config=self.params, # Log all run-specific params
            id=self.run_id,
            resume="allow",
            reinit=True
            )
        best_model_path = None
        try:
            prep = PreprocessingTraining(
                self.raw_text,
                self.tokenizer,
                batch_size=self.params["batch_size"],
                time_steps=self.params["context_window"],
            )
            model = TransformerModel(
                vocab_size=prep.vocab_size,
                channel_dim=self.params["channel_dim"],
                context_window=prep.time_steps,
                num_heads=self.params["num_heads"],
                num_layers=self.params["num_layers"],
            ).to(self.device)
            start_step, opt_state, best_loss = load_checkpoint(model, self.latest_ckpt_path, self.device)

            if start_step > self.params["steps"]:
                logger.warning(f"Run {self.run_id} already completed {start_step - 1} steps. Skipping.")
                print(f"Run {self.run_id} already completed. Skipping.")
                wandb.finish()
                return self.best_ckpt_path # Return best path even if skipped

            optimizer = optim.Adam(model.parameters(), lr=self.params["learning_rate"])
            
            best_model_path = train_loop(
                model,
                prep,
                optimizer,
                self.cfg,
                self.params, # Pass run_params
                self.device,
                start_step,
                opt_state,
                best_loss,
            )

            # Use the actual best_model_path returned by train_loop
            self.best_ckpt_path = best_model_path 

            if self.is_single_run and self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
                run_post_training_eval(model, prep, self.best_ckpt_path, self.cfg, self.device)

            if (self.is_single_run or self.cfg.ALWAYS_LOG_ARTIFACTS) and self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
                log_artifact(self.best_ckpt_path, self.run_id, self.params, self.cfg)

        except Exception as e:
            logger.critical(f"Critical error in run {self.run_id}: {e}", exc_info=True)
            print(f"CRITICAL ERROR in run {self.run_id}. See logs.")
        finally:
            if wandb.run:
                wandb.finish()

        return self.best_ckpt_path


def run_single_training(raw_text, tokenizer, cfg):
    """configures and runs a training run with fixed hyperparameters, ie a single phase"""
    logger.info("--- Starting Single Training Run ---")
    cfg.LEARNING_RATE = float(cfg.LEARNING_RATE)
    run_id = f"single_bs{cfg.BATCH_SIZE}_cw{cfg.CONTEXT_WINDOW}_lr{cfg.LEARNING_RATE:.0e}"
    single_params = {
        "batch_size": cfg.BATCH_SIZE,
        "context_window": cfg.CONTEXT_WINDOW,
        "learning_rate": cfg.LEARNING_RATE,
        "steps": cfg.TRAINING_STEPS,
        "val_check_every": cfg.VALIDATION_CHECK_EVERY,
        "patience": cfg.EARLY_STOPPING_PATIENCE,
        "channel_dim": cfg.CHANNEL_DIM,
        "num_heads": cfg.NUM_HEADS,
        "num_layers": cfg.NUM_LAYERS,
        "name": f"{cfg.WANDB_RUN_PREFIX[0]}_{run_id}", # Added run_id to name
        "notes": "Single training run with standard parameters.",
        "tags": ["single_run", cfg.DATASET_NAME, "transformer"],
        "run_id": run_id,
    }
    manager = TrainingManager(single_params, tokenizer, raw_text, cfg, is_single_run=True)
    manager.run()
    logger.info("--- Single Training Run Finished ---")
    

def run_hyperparameter_search(raw_text, tokenizer, cfg):
    """Configures and runs multiple training sessions for tuning."""
    logger.info("--- Starting Hyperparameter Search ---")
    results = []
    for bs in cfg.HP_SEARCH_BATCH_SIZES:
        for cw in cfg.HP_SEARCH_CONTEXT_WINDOWS:
            for lr in cfg.HP_SEARCH_LRS:
                run_id = f"tune_{bs}_{cw}_{lr:.0e}"
                run_name = f"{cfg.WANDB_RUN_PREFIX[1]}_bs{bs}_cw{cw}_lr{lr:.0e}"

                logger.info(f"Starting Tuning Run: {run_name}")
                tune_params = {
                    "batch_size": bs,
                    "context_window": cw,
                    "learning_rate": lr,
                    "steps": cfg.HP_SEARCH_STEPS,
                    "val_check_every": cfg.HP_VALIDATION_CHECK_EVERY,
                    "patience": cfg.HP_EARLY_STOPPING_PATIENCE,
                    "channel_dim": cfg.CHANNEL_DIM,
                    "num_heads": cfg.NUM_HEADS,
                    "num_layers": cfg.NUM_LAYERS,
                    "name": run_name,
                    "notes": f"Tuning: BS={bs}, CW={cw}, LR={lr}",
                    "tags": ["tuning", cfg.DATASET_NAME, "transformer"],
                    "run_id": run_id,
                }
                manager = TrainingManager(tune_params, tokenizer, raw_text, cfg)
                best_ckpt = manager.run()
                results.append({"params": tune_params, "best_ckpt": best_ckpt})

    logger.info("--- Hyperparameter Search Finished ---")