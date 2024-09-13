# /training/manager.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import os
import logging
import threading
import json
from datetime import datetime

from src.models.transformer import TransformerModel
from src.data.processing import PreprocessingTraining
from src.training.utils import train_loop, load_checkpoint, evaluate_validation_loss

logger = logging.getLogger(__name__)


def save_completion_status(run_id, cfg, completion_info):
    """Saves completion status to a persistent metadata file."""
    completion_dir = os.path.join(cfg.MODEL_CHECKPOINTS_DIR, "completion_tracking")
    os.makedirs(completion_dir, exist_ok=True)
    
    completion_file = os.path.join(completion_dir, f"{run_id}_completed.json")
    
    completion_data = {
        "run_id": run_id,
        "completed": True,
        "completion_timestamp": datetime.now().isoformat(),
        "completion_reason": completion_info.get("reason", "unknown"),
        "final_step": completion_info.get("final_step", None),
        "final_best_loss": completion_info.get("final_best_loss", None),
        "best_checkpoint_logged": completion_info.get("best_checkpoint_logged", False),
        "post_training_eval_done": completion_info.get("post_training_eval_done", False)
    }
    
    try:
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f, indent=2)
        logger.info(f"Saved completion status to {completion_file}")
    except Exception as e:
        logger.error(f"Failed to save completion status: {e}", exc_info=True)


def load_completion_status(run_id, cfg):
    """Loads completion status from persistent metadata file."""
    completion_dir = os.path.join(cfg.MODEL_CHECKPOINTS_DIR, "completion_tracking")
    completion_file = os.path.join(completion_dir, f"{run_id}_completed.json")
    
    if not os.path.exists(completion_file):
        return None
    
    try:
        with open(completion_file, 'r') as f:
            completion_data = json.load(f)
        logger.info(f"Found completion status for run {run_id}: {completion_data['completion_reason']} at step {completion_data.get('final_step', 'unknown')}")
        return completion_data
    except Exception as e:
        logger.error(f"Failed to load completion status for run {run_id}: {e}", exc_info=True)
        return None


def run_post_training_eval(model, prep, best_model_path, cfg, device):
    """Loads the best model and runs test evaluation and text generation."""
    logger.info("Running post-training evaluations...")
    try:
        # Load the best model state
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from {best_model_path} for final eval.")

        # Evaluate on test set
        test_loss = evaluate_validation_loss(
            model, prep, device, "test", cfg.EVAL_ITERS_TEST
        )
        test_loss_str = (
            f"{test_loss:.4f}" if not torch.isnan(torch.tensor(test_loss)) else "N/A"
        )
        logger.info(f"Average Test Set Loss: {test_loss_str}")
        wandb.log({"test_loss_avg": test_loss})

    except Exception as e:
        logger.error(f"Post-training evaluation failed: {e}", exc_info=True)


def log_artifact(checkpoint_path, run_id, run_params, cfg):
    """Logs the best model checkpoint as a W&B Artifact."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found for artifact logging.")
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
        wandb.log_artifact(model_artifact, aliases=["best", run_id])
        logger.info(f"Logged artifact: {artifact_name}")
        return model_artifact
    except Exception as e:
        logger.error(f"Failed to log artifact: {e}", exc_info=True)
        return None  # failure


"""
----------------------------------------------------------------------------------
Why do we use threading for artifact upload timeout?
----------------------------------------------------------------------------------
The W&B artifact.wait() method is a blocking call: it will pause the program until
 the upload is finished, regardless of how long it takes. There is no built-in timeout.

To prevent our training process from hanging indefinitely (e.g., due to network issues
or W&B service problems), we run artifact.wait() in a separate thread. This allows the
main program to wait (join) on that thread for a fixed amount of time (e.g., 300 seconds).

- If the thread finishes within the timeout, the upload is considered successful and we
  can safely clean up local checkpoints.
- If the thread is still running after the timeout, it means the upload is taking too long
  (possibly stuck). We log an error and do NOT delete local checkpoints, to avoid data loss.

Note: We do NOT create a separate thread for the upload itself (W&B handles uploads in the
background). The thread is only for waiting on the blocking wait() call. We do not forcibly
kill the thread if it times out (Python does not support safe thread termination), but we
stop waiting and proceed with error handling.

This approach ensures our pipeline is robust and never hangs forever on artifact upload.
----------------------------------------------------------------------------------
"""


def wait_for_artifact_upload(artifact, timeout_sec=300):
    """
    Waits for the W&B artifact upload to complete, with a timeout.
    Args:
        artifact: The W&B artifact object whose upload we want to wait for.
        timeout_sec: Maximum number of seconds to wait before timing out (default: 300).
    Returns:
        True if upload finished successfully within the timeout, False if timed out or error occurred.
    Why:
        W&B's artifact.wait() is a blocking call and does not support a timeout. In distributed or unreliable network scenarios,
        this can cause the process to hang indefinitely. We use a thread to enforce a timeout for robustness.
    """
    result = {"finished": False}

    def _wait():
        try:
            artifact.wait()
            result["finished"] = True
        except Exception as e:
            logger.error(f"Exception during artifact.wait(): {e}", exc_info=True)

    t = threading.Thread(target=_wait)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        logger.error(
            f"Artifact upload did not finish within {timeout_sec} seconds. Timeout reached."
        )
        return False
    return result["finished"]


class TrainingManager:
    """Manages the complete training session end to end."""

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
        self.latest_ckpt_path = os.path.join(
            self.cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_latest.pt"
        )
        self.best_ckpt_path = os.path.join(
            self.cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_best.pt"
        )

    def _cleanup_checkpoints(self):
        """Removes local checkpoint files after successful upload."""
        logger.info("Cleaning up local checkpoint files...")
        for path in [self.latest_ckpt_path, self.best_ckpt_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removed checkpoint: {path}")
            except OSError as e:
                logger.error(f"Error removing file {path}: {e}", exc_info=True)

    def run(self):
        """Executes the entire training session."""
        wandb.init(
            project=self.cfg.WANDB_PROJECT,
            name=self.params["name"],
            notes=self.params.get("notes", ""),
            tags=self.params.get("tags", []),
            config=self.params,  # Log all run-specific params
            id=self.run_id,
            resume="allow",
            reinit=True,
        )
        
        # Check persistent completion status first
        completion_status = load_completion_status(self.run_id, self.cfg)
        if completion_status and completion_status.get("completed", False):
            logger.warning(
                f"Run {self.run_id} was previously completed ({completion_status['completion_reason']} at step {completion_status.get('final_step', 'unknown')}). Skipping training."
            )
            print(f"Run {self.run_id} was previously completed. Skipping training.")
            
            # For completed runs that were cleaned up, we might need to retrieve the model from W&B artifacts
            # For now, we'll just log that this run was already completed and exit
            if wandb.run:
                wandb.log({"run_status": "already_completed", "completion_reason": completion_status['completion_reason']})
                wandb.finish()
            return None
        
        # Use the predefined best path as the default. It will be updated by train_loop if training occurs.
        final_best_model_path = self.best_ckpt_path
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
                dropout_rate=self.params["dropout_rate"],
                final_dropout_multiplier=self.params["final_dropout_multiplier"],
                max_dropout_val=self.params["max_dropout_val"],
            ).to(self.device)
            
            # Load checkpoint to resume training if it exists
            start_step, opt_state, best_loss, was_completed = load_checkpoint(
                model, self.latest_ckpt_path, self.device
            )

            # The was_completed flag from checkpoint is now only used for runs that haven't been cleaned up yet
            # For cleaned up runs, we rely on the persistent completion status checked above
            if was_completed:  # This handles cases where checkpoint still exists and is marked completed
                logger.warning(
                    f"Run {self.run_id} checkpoint indicates completion. This shouldn't happen if cleanup worked properly."
                )
                print(f"Run {self.run_id} was previously completed (from checkpoint). Skipping training.")
                
                # Save completion status for future reference and clean up
                completion_info = {
                    "reason": "found_completed_checkpoint",
                    "final_step": start_step - 1,
                    "final_best_loss": best_loss
                }
                save_completion_status(self.run_id, self.cfg, completion_info)
                
                # Handle best checkpoint loading
                if os.path.exists(self.best_ckpt_path):
                    best_checkpoint = torch.load(
                        self.best_ckpt_path, map_location=self.device
                    )
                    model.load_state_dict(best_checkpoint["model_state_dict"])
                    final_best_model_path = self.best_ckpt_path
                else:
                    logger.warning(
                        f"Best model checkpoint {self.best_ckpt_path} not found for a completed run."
                    )
                    if wandb.run:
                        wandb.finish()
                    return None
            elif (
                start_step > self.params["steps"]
            ):  # This handles cases where a run was partially completed, but now the desired steps are less than what was already done.
                logger.warning(
                    f"Run {self.run_id} already completed {start_step - 1} steps, which is more than target {self.params['steps']}. Marking as completed."
                )
                print(f"Run {self.run_id} already completed target steps. Marking as completed.")
                
                # Save completion status
                completion_info = {
                    "reason": "exceeded_target_steps",
                    "final_step": start_step - 1,
                    "final_best_loss": best_loss
                }
                save_completion_status(self.run_id, self.cfg, completion_info)
                
                # Handle best checkpoint loading
                if os.path.exists(self.best_ckpt_path):
                    best_checkpoint = torch.load(
                        self.best_ckpt_path, map_location=self.device
                    )
                    model.load_state_dict(best_checkpoint["model_state_dict"])
                    final_best_model_path = self.best_ckpt_path
                else:
                    logger.warning(
                        f"Best model checkpoint {self.best_ckpt_path} not found for a run that exceeded target steps."
                    )
                    if wandb.run:
                        wandb.finish()
                    return None
            else:
                # Build parameter groups for AdamW so that weight-decay (L2 regularisation) is applied **only** to the parameters that benefit from it.
                """
                Why do we split parameters?
                Weight decay (L2 regularization) helps reduce overfitting by shrinking large weights.
                But not all parameters should be regularized! Specifically:

                Bias terms and LayerNorm (or RMSNorm) parameters play stabilizing roles and should NOT be regularized. Including them in weight decay can harm convergence and model capacity.

                Intuition behind exclusions:
                -----------------------------------------------------------------------------
                • Biases (e.g., `*.bias`):
                - Bias adds a constant shift to the activation.
                - It allows the neuron to activate even when inputs are zero.
                - It **does not contribute to model complexity or overfitting** much.
                - Applying weight decay pushes bias toward zero, which reduces model flexibility
                and may prevent certain neurons from activating effectively.
                • LayerNorm weights (e.g., `*.weight` in LayerNorm or RMSNorm):
                - These are 1-D scale vectors (γ) applied after normalizing the activations.
                - They **restore variance** after normalization.
                - If you apply weight decay here, γ gets pulled toward zero,
                which effectively cancels out the normalized signal — destabilizing training.
                • General heuristic:
                Any parameter with `ndim == 1` is typically a scale or bias vector
                Exclude from weight decay.
                """
                decay_params, no_decay_params = [], []
                for name, param in model.named_parameters():
                    if (
                        not param.requires_grad
                    ):  # frozen parameters which wont get trained
                        continue

                    # Anything that is purely multiplicative-scaling or a bias stays out
                    if (
                        param.ndim == 1  # vectors (eg: LayerNorm weight)
                        or name.endswith(".bias")
                        or "layernorm" in name.lower()
                        or "ln" in name.lower()
                    ):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

                # Create the optimizer with the 2 groups
                weight_decay = self.params.get("weight_decay", 0.01)
                optimizer = optim.AdamW(
                    [
                        {"params": decay_params, "weight_decay": weight_decay},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    lr=self.params["learning_rate"],
                )

                # Halve the LR (factor=0.5) if val_loss doesn't improve by >min_delta for 5 validation checks (mode="min" means lower loss is better)

                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=5,
                    threshold=self.cfg.MIN_DELTA,
                )

                training_result = train_loop(
                    model,
                    prep,
                    optimizer,
                    scheduler,
                    self.cfg,
                    self.params,  # Pass run_params
                    self.device,
                    start_step,
                    opt_state,
                    best_loss,
                )
                
                # Extract information from training result
                final_best_model_path = training_result["best_checkpoint_path"]
                training_completed = training_result["completed_successfully"]
                
                # Save completion status if training completed successfully
                if training_completed:
                    completion_info = {
                        "reason": training_result["completion_reason"],
                        "final_step": training_result["final_step"],
                        "final_best_loss": training_result["final_best_loss"]
                    }
                    save_completion_status(self.run_id, self.cfg, completion_info)

            # --- POST-TRAINING LOGIC (APPLIES TO NEWLY COMPLETED AND RESUMED-COMPLETED RUNS) ---
            if not os.path.exists(final_best_model_path):
                logger.warning(
                    f"Best model path '{final_best_model_path}' not found. Skipping post-training steps."
                )
                if (
                    wandb.run
                ):  # Finish wandb if we are exiting early due to missing model
                    wandb.finish()
                return None  # Exit if there is no best model to work with

            # Only run post-training eval if it's a single run
            if self.is_single_run:
                run_post_training_eval(
                    model, prep, final_best_model_path, self.cfg, self.device
                )

            logged_artifact = None
            # Log artifact if it's a single run, or if ALWAYS_LOG_ARTIFACTS is true
            if self.is_single_run or self.cfg.ALWAYS_LOG_ARTIFACTS:
                logged_artifact = log_artifact(
                    final_best_model_path, self.run_id, self.params, self.cfg
                )

            if logged_artifact:  # If artifact was logged successfully on wandb, we can clear up local checkpoints
                try:
                    logger.info(
                        "Waiting for artifact upload to complete (timeout: 300s)..."
                    )
                    # Wait for artifact upload with a timeout to avoid indefinite hanging
                    success = wait_for_artifact_upload(logged_artifact, timeout_sec=300)
                    if success:
                        logger.info("Artifact upload finished.")
                        self._cleanup_checkpoints()
                    else:
                        logger.error(
                            "Artifact upload timed out or failed. Local checkpoints will NOT be deleted. Please check your network or W&B status."
                        )
                except Exception as e:
                    logger.error(
                        f"An error occurred during artifact cleanup: {e}", exc_info=True
                    )

        except Exception as e:
            logger.critical(f"Critical error in run {self.run_id}: {e}", exc_info=True)
            print(f"CRITICAL ERROR in run {self.run_id}. See logs.")
        finally:
            if wandb.run:
                wandb.finish()

        return final_best_model_path


def run_single_training(raw_text, tokenizer, cfg):
    """configures and runs a training run with fixed hyperparameters, ie a single phase"""
    logger.info("--- Starting Single Training Run ---")
    cfg.LEARNING_RATE = float(cfg.LEARNING_RATE)
    run_id = (
        f"single_bs{cfg.BATCH_SIZE}_cw{cfg.CONTEXT_WINDOW}_lr{cfg.LEARNING_RATE:.0e}"
    )
    # base dropout rate
    base_dropout = getattr(cfg, "DROPOUT_RATE", 0.1)  # Default to 0.1 if not in cfg
    final_multiplier = getattr(
        cfg, "FINAL_DROPOUT_MULTIPLIER", None
    )  # Default to None if not in cfg
    max_dropout = getattr(cfg, "MAX_DROPOUT_VAL", 0.5)
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
        "dropout_rate": base_dropout,
        "final_dropout_multiplier": final_multiplier,
        "max_dropout_val": max_dropout,
        "name": f"{cfg.WANDB_RUN_PREFIX[0]}_{run_id}",  # Added run_id to name
        "notes": "Single training run with standard parameters.",
        "tags": ["single_run", cfg.DATASET_NAME, "transformer"],
        "run_id": run_id,
    }
    manager = TrainingManager(
        single_params, tokenizer, raw_text, cfg, is_single_run=True
    )
    manager.run()
    logger.info("--- Single Training Run Finished ---")


def run_hyperparameter_search(raw_text, tokenizer, cfg):
    """Configures and runs multiple training sessions for tuning."""
    logger.info("--- Starting Hyperparameter Search ---")
    results = []
    base_dropout = getattr(cfg, "DROPOUT_RATE", 0.1)  # Default to 0.1 if not in cfg
    final_multiplier = getattr(
        cfg, "FINAL_DROPOUT_MULTIPLIER", None
    )  # Default to None if not in cfg
    max_dropout = getattr(cfg, "MAX_DROPOUT_VAL", 0.5)
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
                    "dropout_rate": base_dropout,
                    "final_dropout_multiplier": final_multiplier,
                    "max_dropout_val": max_dropout,
                    "name": run_name,
                    "notes": f"Tuning: BS={bs}, CW={cw}, LR={lr}",
                    "tags": ["tuning", cfg.DATASET_NAME, "transformer"],
                    "run_id": run_id,
                }
                manager = TrainingManager(tune_params, tokenizer, raw_text, cfg)
                best_ckpt = manager.run()
                results.append({"params": tune_params, "best_ckpt": best_ckpt})

    logger.info("--- Hyperparameter Search Finished ---")


def clear_completion_status(run_id, cfg):
    """Clears completion status for a specific run - useful for debugging or re-running."""
    completion_dir = os.path.join(cfg.MODEL_CHECKPOINTS_DIR, "completion_tracking")
    completion_file = os.path.join(completion_dir, f"{run_id}_completed.json")
    
    if os.path.exists(completion_file):
        try:
            os.remove(completion_file)
            logger.info(f"Cleared completion status for run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear completion status for run {run_id}: {e}", exc_info=True)
            return False
    else:
        logger.info(f"No completion status found for run {run_id}")
        return False


def list_completed_runs(cfg):
    """Lists all completed runs with their completion information."""
    completion_dir = os.path.join(cfg.MODEL_CHECKPOINTS_DIR, "completion_tracking")
    
    if not os.path.exists(completion_dir):
        logger.info("No completion tracking directory found.")
        return []
    
    completed_runs = []
    try:
        for filename in os.listdir(completion_dir):
            if filename.endswith("_completed.json"):
                filepath = os.path.join(completion_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        completion_data = json.load(f)
                    completed_runs.append(completion_data)
                except Exception as e:
                    logger.error(f"Failed to read completion file {filename}: {e}")
                    continue
        
        # Sort by completion timestamp
        completed_runs.sort(key=lambda x: x.get('completion_timestamp', ''), reverse=True)
        return completed_runs
    
    except Exception as e:
        logger.error(f"Failed to list completed runs: {e}", exc_info=True)
        return []


def print_completed_runs_summary(cfg):
    """Prints a summary of all completed runs."""
    completed_runs = list_completed_runs(cfg)
    
    if not completed_runs:
        print("No completed runs found.")
        return
    
    print(f"\nFound {len(completed_runs)} completed runs:")
    print("-" * 80)
    print(f"{'Run ID':<40} {'Reason':<20} {'Final Step':<12} {'Final Loss':<12}")
    print("-" * 80)
    
    for run in completed_runs:
        run_id = run.get('run_id', 'N/A')[:39]  # Truncate if too long
        reason = run.get('completion_reason', 'N/A')[:19]
        final_step = str(run.get('final_step', 'N/A'))[:11]
        final_loss = f"{run.get('final_best_loss', 'N/A'):.4f}" if isinstance(run.get('final_best_loss'), (int, float)) else 'N/A'
        final_loss = final_loss[:11]
        
        print(f"{run_id:<40} {reason:<20} {final_step:<12} {final_loss:<12}")
