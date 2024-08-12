import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import os
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from model import TransformerModel # Import your model
from data_utils import PreprocessingTraining # Import your data utilities

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_validation_loss(model, prep_obj, split = 'validation', eval_iters=20):
    """
    Computes average validation loss over `eval_iters` mini-batches.
    """
    model.eval() #
    losses = [] #
    for _ in range(eval_iters): #
        try: #
            xb, yb = prep_obj.get_batch(split) #
            xb, yb = xb.to(DEVICE), yb.to(DEVICE) #
            _, loss = model(xb, yb) #
            losses.append(loss.item()) #
        except ValueError as e: #
            logger.warning(f"Skipping a batch for split '{split}' during evaluation due to: {e}") #
            continue 
    model.train() #
    return sum(losses) / len(losses) if losses else float('nan') #

def train_model(model, prep_obj, steps=2000, val_check_every=50, patience=4, min_delta=1e-3, lr=5e-3):
    """
    Training loop with early stopping, validation checks, and plotting.
    """
    model.to(DEVICE) #
    logger.info(f"Starting training on {DEVICE}") 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-5) # l2 regularisation
    best_val_loss = float('inf') 
    stale_checks = 0 
    
    checkpoints_dir = "model_checkpoints" #
    os.makedirs(checkpoints_dir, exist_ok=True) #
    current_config = wandb.run.config 
    b_s = current_config.get("batch_size", "N/A") #
    c_w = current_config.get("context_window", "N/A") #
    l_r = current_config.get("learning_rate", "N/A") #
    
    checkpoint_filename = f"transformer_bs{b_s}_cw{c_w}_lr{l_r:.0e}_best.pt" #
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename) #

    train_losses = [] #
    val_loss_dict = {} #
    model.train() #

    for step in range(1, steps + 1): #
        xb, yb = prep_obj.get_batch('train') #
        xb, yb = xb.to(DEVICE), yb.to(DEVICE) #
        logits, loss = model(xb, yb) #

        optimizer.zero_grad() #
        loss.backward() #
        optimizer.step() #

        train_losses.append(loss.item()) #
        wandb.log({"train/loss": loss.item()}, step=step) #

        if step % val_check_every == 0 or step == steps: #
            val_loss = evaluate_validation_loss(model, prep_obj, split='validation', eval_iters=20)
            val_loss_dict[step] = val_loss #
            
            wandb.log({"val/loss": val_loss}, step=step) #

            logger.info(f"[Step {step}/{steps}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}") #
            print(f"[Step {step}/{steps}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}") #


            if val_loss < best_val_loss - min_delta : #
                best_val_loss = val_loss #
                stale_checks = 0 #
                torch.save(model.state_dict(), checkpoint_path) #
                logger.info(f" ↑ New Best Model (Val Loss: {best_val_loss:.4f}); Checkpoint saved to {checkpoint_path}") #
                print(" ↑ New Best Model; Checkpoint saved.") #
                wandb.save(checkpoint_path) #
            else:
                stale_checks += 1 #
                logger.info(f" No improvement for {stale_checks}/{patience} validation checks.") #
                print(f" No improvement for {stale_checks}/{patience} validation checks.") #

            if stale_checks >= patience: #
                logger.warning(">>> Early stopping triggered.") #
                print(">>> Early stopping triggered.") #
                break
    
    if train_losses and val_loss_dict: #
        plt.figure(figsize=(12, 7)) #

        plt.plot(train_losses, label='Training Loss (per step)', color='lightblue', alpha=0.7, linewidth=1) #
        
        if len(train_losses) >= val_check_every: #
            smoothed_train_loss = [sum(train_losses[i:i+val_check_every])/val_check_every for i in range(0, len(train_losses) - val_check_every + 1, val_check_every)] #
            steps_for_smoothed = list(range(val_check_every, len(train_losses)+1, val_check_every))[:len(smoothed_train_loss)] #
            plt.plot(steps_for_smoothed, smoothed_train_loss, label=f'Smoothed Training Loss (avg over {val_check_every} steps)', color='blue', linewidth=2) #

        plt.plot(list(val_loss_dict.keys()), list(val_loss_dict.values()), 'o-', label='Validation Loss', color='orange', linewidth=2, markersize=5) #
        
        plt.xlabel("Training Step") #
        plt.ylabel("Loss") #
        plt.title("Training vs Validation Loss") #
        plt.legend() #
        plt.grid(True, linestyle='--', alpha=0.6) #
        plt.tight_layout() #

        plots_dir = "training_plots" #
        os.makedirs(plots_dir, exist_ok=True) #
        
        plot_filename = f"loss_curve_bs{b_s}_cw{c_w}_lr{l_r:.0e}.png" #
        plot_path = os.path.join(plots_dir, plot_filename) #
        
        plt.savefig(plot_path) #
        wandb.log({"loss_curve": wandb.Image(plot_path)}, step=step) #
        logger.info(f"Loss curve saved to {plot_path} and logged to W&B.") #
    else: #
        logger.info("No data for plotting loss curves.") #

    return train_losses, val_loss_dict #

def hyperparameter_search(raw_text, tokenizer, lrs=[1e-2], batch_sizes=[4], time_steps_list=[8]): #
    results = [] #
    base_model_params = {'num_heads': 8, 'num_layers': 6, 'channel_dim': 64} #

    for bs in batch_sizes: #
        for ts in time_steps_list: #
            for lr in lrs: #
                try: #
                    print(f"Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}") #
                    logger.info(f"Starting Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}") #

                    prep = PreprocessingTraining(raw_text, tokenizer=tokenizer, batch_size=bs, time_steps=ts) #
                    
                    model = TransformerModel( #
                        vocab_size=prep.vocab_size, 
                        context_window=prep.time_steps, 
                        channel_dim=base_model_params['channel_dim'], 
                        num_heads=base_model_params['num_heads'],
                        num_layers=base_model_params['num_layers']
                    ).to(DEVICE) 

                    train_model(model, prep, steps=1000, val_check_every=20, patience=6, lr=lr) 

                    val_loss = evaluate_validation_loss(model, prep, split = 'validation') #
                    results.append({ #
                        'batch_size': bs, #
                        'time_steps': ts, #
                        'learning_rate': lr, #
                        'val_loss': val_loss #
                    })
                    logger.info(f"Completed run: bs={bs}, ts={ts}, lr={lr}, val_loss={val_loss:.4f}") #
                except Exception as e: #
                    logger.error(f"Error in tuning run (bs={bs}, ts={ts}, lr={lr}): {e}", exc_info=True) #
    return results #

if __name__ == '__main__':
    # Main execution for training
    # You'd typically parse arguments here to decide if to train or inference
    pass # This would be called from main.py