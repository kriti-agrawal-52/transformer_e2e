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
import sys

from src.models.transformer import TransformerModel
from src.data.processing import PreprocessingTraining
from src.training.utils import train_loop, load_checkpoint, evaluate_validation_loss, get_checkpoint_paths

logger = logging.getLogger(__name__)



def generate_smart_run_id(base_run_id, cfg):
    """
    Intelligent run ID generation that balances resumption with conflict avoidance.
    
    Logic:
    1. Check if base_run_id was completed → create new run with incremental suffix
    2. If not completed but has checkpoints → resume with same ID (interrupted run)
    3. If no completion status and no checkpoints → use base_run_id (fresh start)
    4. If suffix runs exist, find next available or resume interrupted suffix run
    
    Args:
        base_run_id: Base run identifier (e.g., "single_bs16_cw128_lr1e-04")
        cfg: Configuration object
        
    Returns:
        tuple: (final_run_id, is_resuming_flag)
    """
    
    def has_checkpoints(run_id):
        """Check if checkpoints exist for a given run_id"""
        if not os.path.exists(cfg.MODEL_CHECKPOINTS_DIR):
            return False
            
        checkpoint_paths = get_checkpoint_paths(run_id, cfg)
        
        return os.path.exists(checkpoint_paths['latest']) or os.path.exists(checkpoint_paths['best'])
    
    def is_completed(run_id):
        """Check if a run was already completed"""
        completion_status = load_completion_status(run_id, cfg)
        return completion_status is not None and completion_status.get("completed", False)
    
    # Check base run first
    if is_completed(base_run_id):
        logger.info(f"Run {base_run_id} was already completed. Creating new run with suffix.")
    elif has_checkpoints(base_run_id):
        logger.info(f"Found checkpoints for incomplete run {base_run_id}. Resuming interrupted training.")
        return base_run_id, True
    else:
        # Fresh start - no completion status and no checkpoints
        logger.info(f"No previous data for {base_run_id}. Starting fresh training.")
        return base_run_id, False
    
    # Base run was completed or we need a suffix - find next available
    suffix = 2
    while suffix <= 999:  # Reasonable limit to prevent infinite loops
        candidate_run_id = f"{base_run_id}_{suffix}"
        
        if is_completed(candidate_run_id):
            logger.debug(f"Run {candidate_run_id} was already completed. Checking next suffix.")
            suffix += 1
            continue
        elif has_checkpoints(candidate_run_id):
            logger.info(f"Found checkpoints for incomplete run {candidate_run_id}. Resuming interrupted training.")
            return candidate_run_id, True
        else:
            # Found available suffix
            logger.info(f"Using new run ID: {candidate_run_id}")
            return candidate_run_id, False
    
    # Fallback with timestamp if too many runs (unlikely)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fallback_run_id = f"{base_run_id}_{timestamp}"
    logger.warning(f"Many existing runs found. Using timestamp fallback: {fallback_run_id}")
    return fallback_run_id, False


def add_run_metadata(params, is_resuming=False):
    """
    Add chronological and resumption metadata to run parameters for W&B tracking.
    
    Args:
        params: Dictionary of run parameters
        is_resuming: Boolean indicating if this is a resumed run
        
    Returns:
        Updated params dictionary with metadata
    """
    metadata = {
        "created_timestamp": datetime.now().isoformat(),
        "created_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_resuming": is_resuming,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    # Add metadata to params
    params["run_metadata"] = metadata
    return params


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


def run_post_training_eval(model, prep, best_model_path, run_params, device):
    """Loads the best model and runs test evaluation and text generation."""
    logger.info("Running post-training evaluations...")
    try:
        # Load the best model state
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from {best_model_path} for final eval.")

        # Extract system-level configs from run_params
        cfg = run_params["cfg"]
        
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


def log_artifact(checkpoint_path, run_id, run_params):
    """Logs the best model checkpoint as a W&B Artifact."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found for artifact logging.")
        return

    # Extract system-level configs from run_params
    cfg = run_params["cfg"]
    
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

    def __init__(self, run_params, tokenizer, raw_text, cfg):
        self.params = run_params
        # Include cfg in run_params for system-level configurations
        self.params["cfg"] = cfg
        self.tokenizer = tokenizer
        self.raw_text = raw_text
        self.cfg = cfg
        self.run_id = self.params["run_id"]
        self.device = torch.device(cfg.DEVICE)

        checkpoint_paths = get_checkpoint_paths(self.run_id, self.cfg, self.params)
        self.latest_ckpt_path = checkpoint_paths['latest']
        self.best_ckpt_path = checkpoint_paths['best']

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
        """Executes the entire training session with enhanced metadata logging."""
        # Log enhanced metadata and resumption status
        is_resuming = self.params.get("run_metadata", {}).get("is_resuming", False)
        logger.info(f"{'Resuming' if is_resuming else 'Starting'} training run: {self.run_id}")
        
        wandb.init(
            project=self.cfg.WANDB_PROJECT,
            name=self.params["name"],
            notes=self.params.get("notes", ""),
            tags=self.params.get("tags", []),
            config=self.params,  # Log all run-specific params including metadata
            id=self.run_id,
            resume="allow"
        )
        
        # Log creation info for better traceability
        if "run_metadata" in self.params:
            metadata = self.params["run_metadata"]
            logger.info(f"Run created at: {metadata.get('created_readable', 'unknown')}")
            logger.info(f"{'Resuming' if metadata.get('is_resuming', False) else 'Starting fresh'} - Python {metadata.get('python_version', 'unknown')}")
        
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
            start_step, opt_state, best_loss = load_checkpoint(
                model, self.latest_ckpt_path, self.device
            )

            # Check if target steps have already been exceeded
            if (
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
                    threshold=self.params["cfg"].MIN_DELTA,
                )

                training_result = train_loop(
                    model,
                    prep,
                    optimizer,
                    scheduler,
                    self.params,  # Pass run_params (contains cfg and all hyperparameters)
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
            if self.params.get("run_metadata", {}).get("is_resuming", False):
                run_post_training_eval(
                    model, prep, final_best_model_path, self.params, self.device
                )

            logged_artifact = None
            # Log artifact if it's a single run, or if ALWAYS_LOG_ARTIFACTS is true
            if self.params.get("run_metadata", {}).get("is_resuming", False) or self.cfg.ALWAYS_LOG_ARTIFACTS:
                logged_artifact = log_artifact(
                    final_best_model_path, self.run_id, self.params
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

            # Handle training completion and cleanup
            if completion_info and completion_info.get("completed_successfully"):
                completion_reason = completion_info.get("completion_reason", "unknown")
                final_step = completion_info.get("final_step", "unknown")
                
                logger.info(f"Training completed successfully ({completion_reason}) at step {final_step}")
                
                # Save persistent completion status
                persistent_completion_info = {
                    "completed": True,
                    "completion_reason": completion_reason,
                    "final_step": final_step,
                    "completed_at": datetime.now().isoformat(),
                    "best_val_loss": completion_info.get("final_best_loss"),
                }
                save_completion_status(self.run_id, self.cfg, persistent_completion_info)
                
                # Clean up local checkpoints for completed runs
                # This enables the smart resumption logic to work correctly:
                # - If checkpoints exist → interrupted run (should resume)
                # - If no checkpoints exist → completed run (should create new run with suffix)
                try:
                    if os.path.exists(self.latest_ckpt_path):
                        os.remove(self.latest_ckpt_path)
                        logger.info(f"Cleaned up latest checkpoint: {self.latest_ckpt_path}")
                        
                    if os.path.exists(self.best_ckpt_path):
                        os.remove(self.best_ckpt_path) 
                        logger.info(f"Cleaned up best checkpoint: {self.best_ckpt_path}")
                        
                    logger.info("Local checkpoints cleaned up after successful completion.")
                   
                except Exception as e:
                    logger.warning(f"Failed to clean up local checkpoints: {e}")
                    # Don't fail the run due to cleanup issues
            else:
                logger.info("Training did not complete successfully - preserving checkpoints for resumption.")

            # Log final completion status to W&B
            if wandb.run:
                final_status = "completed_successfully" if completion_info and completion_info.get("completed_successfully") else "interrupted"
                completion_time = datetime.now()
                wandb.log({
                    "final_run_status": final_status,
                    "final_step": completion_info.get("final_step") if completion_info else last_step_reached,
                    "completion_time_unix": completion_time.timestamp(),  # Unix timestamp for easy sorting/filtering
                    "completion_time_readable": completion_time.strftime("%Y-%m-%d %H:%M:%S")  # Human readable
                })

        except Exception as e:
            logger.critical(f"Critical error in run {self.run_id}: {e}", exc_info=True)
            print(f"CRITICAL ERROR in run {self.run_id}. See logs.")
        finally:
            if wandb.run:
                wandb.finish()

        return final_best_model_path


def run_single_training(raw_text, tokenizer, cfg):
    """Initializes and runs a single training session."""
    # Create descriptive base_run_id with hyperparameters (like hyperparameter search does)
    base_run_id = f"single_bs{cfg.BATCH_SIZE}_cw{cfg.CONTEXT_WINDOW}_lr{cfg.LEARNING_RATE:.0e}"
    run_id, is_resuming = generate_smart_run_id(base_run_id, cfg)
    
    # Base parameters for this run
    run_params = {
        "batch_size": cfg.BATCH_SIZE,
        "context_window": cfg.CONTEXT_WINDOW,
        "learning_rate": float(cfg.LEARNING_RATE),  # Ensure it's always a float
        "steps": cfg.TRAINING_STEPS,
        "val_check_every": cfg.VALIDATION_CHECK_EVERY,
        "patience": cfg.EARLY_STOPPING_PATIENCE,
        "channel_dim": cfg.CHANNEL_DIM,
        "num_heads": cfg.NUM_HEADS,
        "num_layers": cfg.NUM_LAYERS,
        "dropout_rate": getattr(cfg, "DROPOUT_RATE", 0.1),
        "final_dropout_multiplier": getattr(cfg, "FINAL_DROPOUT_MULTIPLIER", None),
        "max_dropout_val": getattr(cfg, "MAX_DROPOUT_VAL", 0.5),
        "name": f"{cfg.WANDB_RUN_PREFIX[0]}_{run_id}",
        "notes": f"Single training run ({'resumed' if is_resuming else 'new'}) with standard parameters.",
        "tags": ["single_run", cfg.DATASET_NAME, "transformer"] + (["resumed"] if is_resuming else ["fresh_start"]),
        "run_id": run_id,
    }
    
    # Add metadata for W&B
    run_params = add_run_metadata(run_params, is_resuming)

    # Initialize and run the training manager
    manager = TrainingManager(run_params, tokenizer, raw_text, cfg)
    manager.run()


def run_hyperparameter_search(raw_text, tokenizer, cfg):
    """Configures and runs multiple training sessions for tuning with smart resumption logic."""
    logger.info("--- Starting Hyperparameter Search ---")
    results = []
    
    # Generate a sweep timestamp for grouping related runs
    sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # YYYYMMDD_HHMM format
    logger.info(f"Hyperparameter sweep started at: {sweep_timestamp}")
    
    base_dropout = getattr(cfg, "DROPOUT_RATE", 0.1)  # Default to 0.1 if not in cfg
    final_multiplier = getattr(
        cfg, "FINAL_DROPOUT_MULTIPLIER", None
    )  # Default to None if not in cfg
    max_dropout = getattr(cfg, "MAX_DROPOUT_VAL", 0.5)
    
    for bs in cfg.HP_SEARCH_BATCH_SIZES:
        for cw in cfg.HP_SEARCH_CONTEXT_WINDOWS:
            for lr in cfg.HP_SEARCH_LRS:
                # Ensure lr is a float for string formatting
                lr = float(lr)
                
                # Generate base run ID for this hyperparameter combination
                base_run_id = f"tune_bs{bs}_cw{cw}_lr{lr:.0e}"
                
                # Use smart run ID generation (but for sweeps, usually create new runs)
                # We temporarily update cfg for checkpoint checking
                original_cfg_values = {}
                for attr in ['BATCH_SIZE', 'CONTEXT_WINDOW', 'LEARNING_RATE']:
                    if hasattr(cfg, attr):
                        original_cfg_values[attr] = getattr(cfg, attr)
                
                # Temporarily set cfg values for checkpoint checking
                cfg.BATCH_SIZE = bs
                cfg.CONTEXT_WINDOW = cw  
                cfg.LEARNING_RATE = lr
                
                run_id, is_resuming = generate_smart_run_id(base_run_id, cfg)
                
                # Restore original cfg values
                for attr, value in original_cfg_values.items():
                    setattr(cfg, attr, value)
                
                run_name = f"{cfg.WANDB_RUN_PREFIX[1]}_{run_id}"
                logger.info(f"Starting Tuning Run: {run_name} ({'resumed' if is_resuming else 'new'})")
                
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
                    "notes": f"Tuning ({'resumed' if is_resuming else 'new'}): BS={bs}, CW={cw}, LR={lr}",
                    "tags": ["tuning", cfg.DATASET_NAME, "transformer", f"sweep_{sweep_timestamp}"] + (["resumed"] if is_resuming else ["fresh_start"]),
                    "run_id": run_id,
                    "sweep_timestamp": sweep_timestamp,  # Group related sweep runs
                }
                
                # Add chronological and resumption metadata
                tune_params = add_run_metadata(tune_params, is_resuming)
                
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
