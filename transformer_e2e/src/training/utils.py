# training/utils.py
import torch
import wandb
import os
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_checkpoint(step, model, optimizer, best_loss, path, completed = False):
    """Saves a checkpoint containing model/optimizer states, step and training completion status"""
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_loss,
            "completed": completed,
        },
        path,
    )
    logger.info(f"Checkpoint saved to {path} at step {step}")


def load_checkpoint(model, checkpoint_path, device):
    """Loads a checkpoint and returns the start step, optimizer state, best loss, and completion status"""
    start_step = 1
    opt_state = None
    best_loss = float("inf")
    was_completed = False  # Default to false
    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint: {checkpoint_path}. Attempting to resume.")
        print(f"Resuming from checkpoint: {os.path.basename(checkpoint_path)}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            opt_state = checkpoint["optimizer_state_dict"]
            start_step = checkpoint["step"] + 1
            best_loss = checkpoint.get("best_val_loss", float("inf"))
            was_completed = checkpoint.get("completed", False) # Load completion status
            logger.info(
                f"Resuming from step {start_step} "
                f"with best_val_loss {best_loss:.4f}"
                f"{' (Previously completed)' if was_completed else ''}" # Log if completed
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

    return start_step, opt_state, best_loss, was_completed


def plot_and_log_loss(
    train_losses, val_loss_dict, val_check_every, last_step, b_s, c_w, l_r, cfg, run_id
):
    """Helper to plot loss curves and log the image to W&B."""
    if not train_losses or not val_loss_dict:
        return
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
    )  # this ensures that plots for various hyperparameter search runs are saved separately and are not overwritten
    plt.savefig(plot_path)

    wandb.log(
        {"loss_curve": wandb.Image(plot_path)}, step=last_step
    )  # logging figures to weights and biases registry
    logger.info(f"Loss curve saved to {plot_path} and logged.")
    plt.close()


@torch.no_grad()
def evaluate_validation_loss(
    model, prep_obj, device, split="validation", eval_iters=20
):
    """Computes the average loss over several (eval_iters) batches for a given split.
    So, if eval is called val_check_every steps in train_loop,
    then at those steps, we take eval_iters random batches.
    get their losses, and then average them."""
    model.eval()  # puts model to eval mode
    losses = []
    for _ in range(eval_iters):
        try:
            xb, yb = prep_obj.get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        except ValueError as e:
            logger.warning(f"Skipping batch '{split}': {e}")
            continue
    model.train()  # returns model to train mode
    return sum(losses) / len(losses) if losses else float("nan")


def train_loop(
    model,
    prep_obj,
    optimizer,
    scheduler,
    cfg,
    run_params,  # pass the specific run's parameters
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

    # get run-specific config from wandb.run.config
    # this works whether we are in the single run or a hyperparameter search run
    # current_config = wandb.run.config
    b_s = run_params["batch_size"]
    c_w = run_params["context_window"]
    l_r = run_params["learning_rate"]
    steps = run_params["steps"]
    val_check_every = run_params["val_check_every"]
    patience = run_params["patience"]

    # Define checkpoint paths
    # This ensures that checkpoints during runs do not get overwritten when we are doing hyperparameter search
    # this also helps us differentiate between best model during training, and checkpoints saved every few steps to help restart training
    base_filename = f"run_{run_id}_bs{b_s}_cw{c_w}_lr{l_r:.0e}"
    latest_checkpoint_path = os.path.join(
        cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_latest.pt"
    )
    best_checkpoint_path = os.path.join(
        cfg.MODEL_CHECKPOINTS_DIR, f"{base_filename}_best.pt"
    )

    train_losses = []
    val_loss_dict = {}
    model.train()

    training_completed_successfully = False  # Flag to track if training finished without interruption
    
    # Single run training loop
    for step in range(
        start_step, steps + 1
    ):  # we have defined it this way that in case our training run gets interrupted in between
        # , we can load the latest checkpoint and start from start_step
        try:
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

            # Validation and checkpointing logic
            if step % val_check_every == 0 or step == steps:
                val_loss = evaluate_validation_loss(
                    model, prep_obj, device, "validation", cfg.EVAL_ITERS_VAL
                )
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

                save_checkpoint(
                    step, model, optimizer, best_val_loss, latest_checkpoint_path, completed = False
                )  # we are checkpointing the model here

                if val_loss < best_val_loss - cfg.MIN_DELTA:
                    best_val_loss = val_loss
                    stale_checks = 0
                    save_checkpoint(
                        step, model, optimizer, best_val_loss, best_checkpoint_path, completed = False  # Best checkpoint not necessarily final completion
                    )  # saved in the best_checkpoint path as the new best model during training
                    logger.info(f" New Best Model (Val: {best_val_loss:.4f})")
                    print(" New Best Model; Checkpoint saved.")
                    wandb.save(best_checkpoint_path)
                    # wandb.save() uploads the file to the current W&B run's file storage.
                    # This makes the file available for download from the run page, but does not
                    # register it in the W&B Model Registry as an artifact.
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
                    training_completed_successfully = True # Mark as completed on early stop
                    break  # Exit the loop
        except Exception as e:
            logger.error(f"Error during step {step}: {e}", exc_info=True)
            print(f"ERROR during step {step}. Saving latest checkpoint and exiting.")
            # Save latest checkpoint on error, but not as completed
            save_checkpoint(
                step, model, optimizer, best_val_loss, latest_checkpoint_path, completed = False
            )
            raise  # Re-raise to stop
    
    # After the loop (either completed all steps or early stopped)
    else: # This 'else' block executes if the loop completes *without* a 'break' (i.e., all steps finished)
        training_completed_successfully = True
        
    # Save the final latest checkpoint with completion status
    if training_completed_successfully:
        save_checkpoint(
            step, model, optimizer, best_val_loss, latest_checkpoint_path, completed=True
        )
        # If best_checkpoint_path exists and training was successful,
        # ensure it's also marked as completed if it was the last checkpoint
        if os.path.exists(best_checkpoint_path):
            best_checkpoint_data = torch.load(best_checkpoint_path)
            best_checkpoint_data['completed'] = True
            torch.save(best_checkpoint_data, best_checkpoint_path)
    
    

    # Plot loss curves at the end
    plot_and_log_loss(
        train_losses, val_loss_dict, val_check_every, step, b_s, c_w, l_r, cfg, run_id
    )
    # returns the best checkpoint path after training run is complete.
    return best_checkpoint_path
