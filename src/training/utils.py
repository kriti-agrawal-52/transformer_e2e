# training/utils.py
import torch
import wandb
import os
import logging
import matplotlib.pyplot as plt
from src.utils.exceptions import ValidationLossComputationError


logger = logging.getLogger(__name__)


def get_checkpoint_paths(run_id, cfg, run_params=None):
    """
    Centralized utility function to generate checkpoint file paths.
    
    Args:
        run_id: The run identifier
        cfg: Configuration object containing MODEL_CHECKPOINTS_DIR
        run_params: Optional dictionary with run parameters. If provided, 
                   hyperparameters are extracted from here. Otherwise, they're 
                   extracted from cfg.
    
    Returns:
        dict: Dictionary with 'latest' and 'best' checkpoint paths
    """
    if run_params:
        # Extract from run_params (used by TrainingManager)
        bs = run_params["batch_size"]
        cw = run_params["context_window"]
        lr = run_params["learning_rate"]
    else:
        # Extract from cfg (used by generate_smart_run_id)
        bs = cfg.BATCH_SIZE if hasattr(cfg, 'BATCH_SIZE') else "unknown"
        cw = cfg.CONTEXT_WINDOW if hasattr(cfg, 'CONTEXT_WINDOW') else "unknown"
        
        # Handle learning rate - it might be string or numeric in cfg
        if hasattr(cfg, 'LEARNING_RATE'):
            lr_val = cfg.LEARNING_RATE
            # If it's already a string, use it as-is, otherwise format as scientific notation
            if isinstance(lr_val, str):
                lr = lr_val
            else:
                lr = f"{lr_val:.0e}"
        else:
            lr = "unknown"
    
    # Format learning rate consistently
    if isinstance(lr, (int, float)):
        lr_str = f"{lr:.0e}"
    else:
        lr_str = str(lr)
    
    base_filename = f"run_{run_id}_bs{bs}_cw{cw}_lr{lr_str}"
    
    return {
        'latest': os.path.join(cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_latest.pt"),
        'best': os.path.join(cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_best.pt"),
        'base_filename': base_filename  # Also return base filename for other uses
    }


def save_checkpoint(step, model, optimizer, best_loss, path):
    """Saves a checkpoint containing model/optimizer states and step"""
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_loss,
        },
        path,
    )
    logger.info(f"Checkpoint saved to {path} at step {step}")


def load_checkpoint(model, checkpoint_path, device):
    """Loads a checkpoint and returns the start step, optimizer state, and best loss."""
    start_step = 1
    opt_state = None
    best_loss = float("inf")
    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint: {checkpoint_path}. Attempting to resume.")
        print(f"Resuming from checkpoint: {os.path.basename(checkpoint_path)}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            opt_state = checkpoint["optimizer_state_dict"]
            start_step = checkpoint["step"] + 1
            best_loss = checkpoint.get("best_val_loss", float("inf"))
            logger.info(
                f"Resuming from step {start_step} "
                f"with best_val_loss {best_loss:.4f}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load checkpoint {checkpoint_path}: {e}. Starting fresh.",
                exc_info=True,
            )
            print(f"WARN: Failed to load checkpoint {checkpoint_path}. Starting fresh.")
            return 1, None, float("inf")  # Reset on failure
    else:
        logger.info("No checkpoint found. Starting fresh.")

    return start_step, opt_state, best_loss


def plot_and_log_loss(
    train_losses, val_loss_dict, run_params, last_step, run_id
):
    """Helper to plot loss curves and log the image to W&B."""
    if not train_losses or not val_loss_dict:
        return
    
    # Extract necessary parameters from run_params
    val_check_every = run_params["val_check_every"]
    b_s = run_params["batch_size"]
    c_w = run_params["context_window"]
    l_r = run_params["learning_rate"]
    cfg = run_params["cfg"]  # System-level configs like directories
    
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label="Training Loss", color="lightblue", alpha=0.7, lw=1)

    if len(train_losses) >= val_check_every:
        smoothed = [
            sum(train_losses[i : i + val_check_every]) / val_check_every
            for i in range(0, len(train_losses) - val_check_every + 1, val_check_every)
        ]
        steps_s = list(range(val_check_every, len(train_losses) + 1, val_check_every))[
            : len(smoothed)
        ]
        plt.plot(
            steps_s,
            smoothed,
            label=f"Smoothed Training ({val_check_every})",
            color="blue",
            lw=2,
        )
        # The 'smoothed' variable is calculated by taking a non-overlapping block average of the 'train_losses'.
        # It computes the average of 'val_check_every' consecutive loss values for each block.
        # This creates a series of averaged losses, effectively smoothing out fluctuations.
        # matplotlib.pyplot.plot() then connects these block-averaged points with straight lines,
        # where each point represents the average loss of a block and is plotted at the end of that block's steps.

    plt.plot(
        list(val_loss_dict.keys()),
        list(val_loss_dict.values()),
        "o-",
        label="Validation Loss",
        color="orange",
        lw=2,
        ms=5,
    )
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(
        cfg.LOSS_PLOT_DIRECTORY, f"loss_run_{run_id}_bs{b_s}_cw{c_w}_lr{l_r:.0e}.png"
    )
    plt.savefig(plot_path)

    wandb.log(
        {"loss_curve": wandb.Image(plot_path)}, step=last_step
    )  # logging figures to weights and biases registry
    logger.info(f"Loss curve saved to {plot_path} and logged.")
    plt.close()


@torch.no_grad()
def evaluate_validation_loss(
    model, prep_obj, device, split="validation", eval_iters=20, min_successful_ratio=0.5
):
    """
    Computes the average loss over several (eval_iters) batches for a given split.
    If too many batches fail, raises ValidationLossComputationError.
    Args:
        model: The model to evaluate.
        prep_obj: Data preparation object.
        device: Device to run on.
        split: Data split to use.
        eval_iters: Number of batches to evaluate.
        min_successful_ratio: Minimum ratio of successful batches required (e.g., 0.5 for 50%).
    Returns:
        Average loss over successful batches.
    Raises:
        ValidationLossComputationError: If not enough batches succeed.
    """
    model.eval()
    losses = []
    failed_batches = 0
    for _ in range(eval_iters):
        try:
            xb, yb = prep_obj.get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        except ValueError as e:
            # Log and count failed batch
            logger.warning(f"Skipping batch '{split}': {e}")
            failed_batches += 1
            continue
    model.train()

    successful_batches = len(losses)
    min_required = int(eval_iters * min_successful_ratio)

    # If not enough batches succeeded, raise an error
    if successful_batches < min_required:
        raise ValidationLossComputationError(
            f"Validation loss computation failed: only {successful_batches}/{eval_iters} "
            f"batches succeeded (minimum required: {min_required})."
        )

    # If some batches failed but enough succeeded, log a warning
    if successful_batches < eval_iters:
        logger.warning(
            f"Validation loss computed from only {successful_batches}/{eval_iters} batches. "
            "Results may be less reliable."
        )

    return sum(losses) / successful_batches


def train_loop(
    model,
    prep_obj,
    optimizer,
    scheduler,
    run_params,  # Single source of truth for all hyperparameters
    device,
    start_step=1,
    initial_optimizer_state=None,
    initial_best_loss=float("inf"),
):
    """Executes the training loop with validation, stopping, and checkpoints."""
    model.to(device)  # Move model to device
    logger.info(f"Starting/Resuming training on {device}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-4)  # added weight decay for L2 regularization
    if (
        initial_optimizer_state
    ):  # loads the state dict for optimizer if training was interrupted
        optimizer.load_state_dict(initial_optimizer_state)
        logger.info("Loaded optimizer state from checkpoint.")

    best_val_loss = initial_best_loss
    stale_checks = 0
    run_id = wandb.run.id  # use current wandb id

    # Extract all hyperparameters from run_params (single source of truth)
    steps = run_params["steps"]
    val_check_every = run_params["val_check_every"]
    patience = run_params["patience"]
    
    # System-level configurations (extracted once from cfg passed in run_params)
    cfg = run_params["cfg"]  # System configs like directories
    eval_iters_val = cfg.EVAL_ITERS_VAL
    min_successful_val_batch_ratio = cfg.MIN_SUCCESSFUL_VAL_BATCH_RATIO
    min_delta = cfg.MIN_DELTA

    # Define checkpoint paths
    # This ensures that checkpoints during runs do not get overwritten when we are doing hyperparameter search
    # this also helps us differentiate between best model during training, and checkpoints saved every few steps to help restart training
    checkpoint_paths = get_checkpoint_paths(run_id, cfg, run_params)
    latest_checkpoint_path = checkpoint_paths['latest']
    best_checkpoint_path = checkpoint_paths['best']

    train_losses = []
    val_loss_dict = {}
    model.train()

    training_completed_successfully = (
        False  # Flag to track if training finished without interruption
    )
    last_step = start_step - 1

    # Single run training loop
    for step in range(
        start_step, steps + 1
    ):  # we have defined it this way that in case our training run gets interrupted in between
        # , we can load the latest checkpoint and start from start_step
        try:
            last_step = step
            # Get batch and perform training step
            xb, yb = prep_obj.get_batch("train")
            xb, yb = xb.to(device), yb.to(device)  # Move batch to device
            _, loss = model(
                xb, yb
            )  # we forward the batch through the entire model architecture,
            # and get logits and losss, for training, logits are irrelevalt
            optimizer.zero_grad()  # remove old gradients
            loss.backward()  # backpropagration to get step values
            optimizer.step()  # improve params with step

            train_losses.append(
                loss.item()
            )  # append loss value (without gradient attached) to the train_losses list
            wandb.log({"train/loss": loss.item()}, step=step)

            # --- Validation and Checkpointing Logic ---
            # We perform validation at specified intervals and on the final training step.
            is_val_step = step % val_check_every == 0
            is_last_step = step == steps

            if is_val_step or is_last_step:
                try:
                    val_loss = evaluate_validation_loss(
                        model,
                        prep_obj,
                        device,
                        split="validation",
                        eval_iters=eval_iters_val,
                        min_successful_ratio=min_successful_val_batch_ratio,
                    )
                except ValidationLossComputationError as e:
                    logger.critical(f"Validation failed: {e}")
                    print(f"Validation failed: {e}")
                    # Interrupt training
                    raise
                val_loss_dict[step] = val_loss
                wandb.log({"val/loss": val_loss}, step=step)

                # Log current learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"learning_rate": current_lr}, step=step)

                logger.info(
                    f"[Step {step}/{steps}] Train: {loss.item():.4f}, Val: {val_loss:.4f}"
                )
                print(
                    f"[Step {step}/{steps}] Train: {loss.item():.4f}, Val: {val_loss:.4f}"
                )

                # We save a 'latest' checkpoint at each validation step to support resumption.
                # However, we skip this on the final step because a definitive checkpoint
                # will be created after the loop finishes, making this one redundant.
                if not is_last_step:
                    save_checkpoint(
                        step,
                        model,
                        optimizer,
                        best_val_loss,
                        latest_checkpoint_path,
                    )

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    stale_checks = 0
                    # If the model has improved, we save its state as the new 'best' checkpoint.
                    save_checkpoint(
                        step,
                        model,
                        optimizer,
                        best_val_loss,
                        best_checkpoint_path,
                    )
                    logger.info(f" New Best Model (Val: {best_val_loss:.4f})")
                    print(" New Best Model; Checkpoint saved.")
                    # Note: We don't use wandb.save() here as it can cause symlink issues on Google Colab/Drive.
                    # The model will be properly logged as a W&B artifact after training completes.
                else:
                    stale_checks += 1
                    logger.info(f" No improvement: {stale_checks}/{patience}")
                    print(f" No improvement: {stale_checks}/{patience}")

                # Update learning rate scheduler
                # The scheduler automatically tracks its own patience counter (separate from early stopping).
                # It counts how many validation steps pass without improvement and halves the LR
                # when its patience (5 steps) is exceeded. We just call step() and it handles everything.
                scheduler.step(val_loss)

                # Early stopping check
                # we have waited for threshold number of validations (patience) and model has not shown considerable improvement
                if stale_checks >= patience:
                    logger.warning(">>> Early stopping triggered.")
                    print(">>> Early stopping triggered.")
                    training_completed_successfully = (
                        True  # Mark as completed on early stop
                    )
                    break  # Exit the loop
        except Exception as e:
            logger.error(
                f"Training loop interrupted at step {step} due to: {e}",
                exc_info=True,
            )
            print(f"\nERROR: Training interrupted. See logs for details.")
            break  # Exit loop on error
    else:  # This block runs only if the loop completes without a 'break'
        training_completed_successfully = True

    if training_completed_successfully:
        logger.info(
            f"Training completed at step {last_step}. Saving final checkpoint."
        )

        # After a successful training run, we save the final state to the 'latest' checkpoint.
        save_checkpoint(
            last_step,
            model,
            optimizer,
            best_val_loss,
            latest_checkpoint_path,
        )

        # Ensure the best checkpoint is preserved with its final state.
        if os.path.exists(best_checkpoint_path):
            # Load and resave to ensure consistency.
            best_ckpt = torch.load(best_checkpoint_path)
            torch.save(best_ckpt, best_checkpoint_path)
            logger.info(
                f"Preserved best checkpoint at {best_checkpoint_path}."
            )

    # Plot and log loss with simplified parameters
    plot_and_log_loss(
        train_losses,
        val_loss_dict,
        run_params,
        last_step,
        run_id,
    )
    
    # Return training completion information
    completion_info = {
        "best_checkpoint_path": best_checkpoint_path,
        "completed_successfully": training_completed_successfully,
        "final_step": last_step,
        "final_best_loss": best_val_loss,
        "completion_reason": "early_stopping" if stale_checks >= patience else ("all_steps_completed" if training_completed_successfully else "interrupted")
    }
    
    return completion_info
