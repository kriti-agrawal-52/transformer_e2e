import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import logging
import os
from transformers import AutoTokenizer


# Setting up logger
logging.basicConfig(
    filename='transformer_model_wordseq.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, context_window):
        super().__init__()
        
        # Each token will be represented by a vector of size `embed_dim`
        # We'll split it into `num_heads` smaller heads to capture attention from different perspectives
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        # The assert statement is used to enforce that a condition must be true at runtime.
        # If the condition fails, Python raises an AssertionError 
        
        self.num_heads = num_heads                 # Total number of heads
        self.head_dim = embed_dim // num_heads     # Size of each individual head (D)

        # Learnable linear projections for queries, keys, and values
        # They all map from (embed_dim → embed_dim), internally split later into heads
        self.query = nn.Linear(embed_dim, embed_dim, bias = False)
        self.key = nn.Linear(embed_dim, embed_dim, bias = False)
        self.value = nn.Linear(embed_dim, embed_dim, bias = False)

        # After attention, we combine the heads and project back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Lower triangular matrix for causal masking (ensures token i only attends to tokens ≤ i)
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))

    def forward(self, x):
        # x: shape (B, T, C) where:
        # B = batch size, T = time/context length, C = embedding dimension

        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # === Step 1: Linear projections ===
        # Output shape: (B, T, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # === Step 2: Reshape into multi-heads ===
        # From (B, T, C) → (B, H, T, D)
        Q = Q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        # === Step 3: Compute attention scores ===
        # Dot product: Q @ K^T → (B, H, T, T)
        attn_scores = Q @ K.transpose(-2, -1)

        # Scale scores by sqrt(D) for gradient stability
        attn_scores = attn_scores / (D ** 0.5)

        # Apply causal mask (prevent attending to future tokens)
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax across last dimension to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # === Step 4: Compute weighted sum of values ===
        # Weighted sum: (B, H, T, T) @ (B, H, T, D) → (B, H, T, D)
        attn_output = attn_weights @ V

        # === Step 5: Combine heads ===
        # Reshape: (B, H, T, D) → (B, T, H*D = C)
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        #attn_output first is transposed, ie dimensions at index 1 and 2 are swapped (H & T), so shape after transpose : (B, T, H, D)
        # after transpose, the tensor's memory layout might become non-contigouous, elements that are logically adjacent in the new shape might not be physically adjacent in memory
        # required because view pytorch operation requires tensor to be contiguous, otherwise we will get runtime error

        # Final output projection
        return self.out_proj(out)
    
# === Preprocessing Class ===
class PreprocessingTraining():
    def __init__(self, text, tokenizer, batch_size=4, time_steps=64):
        self.text = text
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.time_steps = time_steps
        
        logger.info("Tokenizing the entire dataset...")
        
        self.all_token_ids = self.tokenizer.encode(self.text) # it tokeizes and encodes are input text, so even repeated tokens such as hello hello would be encoded separately.
        logger.info(f"Tokenization complete. Total tokens: {len(self.all_token_ids)}")
        
        # KEY CHANGE: Get vocab size from the tokenizer
        self.vocab_size = self.tokenizer.vocab_size # vocab_size is an attribute of tokenizer object
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        # KEY CHANGE: Split tokenized data
        self._split_tokenized_data() # New method to split token IDs
    
    def _split_tokenized_data(self, train_val_ratio=0.9, val_ratio_of_train_val=0.1):
        """
        Splits `self.all_token_ids` into train, validation, and test sets (as tensors).
        The logic mirrors the original script's 81% train, 9% val, 10% test split.
        """
        n_tokens = len(self.all_token_ids) # number of tokens in the input text
        
        # Determine split point for test data (10% for test)
        train_val_idx_end = int(n_tokens * train_val_ratio) # First 90% for train+val
        
        all_tokens_tensor = torch.tensor(self.all_token_ids, dtype=torch.long)

        train_val_data_tokens = all_tokens_tensor[:train_val_idx_end]
        self.test_tokens = all_tokens_tensor[train_val_idx_end:]

        # Split train_val_data further into train and validation
        # (90% of train_val for train, 10% of train_val for val)
        val_idx_start = int(len(train_val_data_tokens) * (1 - val_ratio_of_train_val))
        
        self.train_tokens = train_val_data_tokens[:val_idx_start]
        self.val_tokens = train_val_data_tokens[val_idx_start:]

        logger.info(f"Train tokens: {len(self.train_tokens)}, Val tokens: {len(self.val_tokens)}, Test tokens: {len(self.test_tokens)}")
        if len(self.train_tokens) == 0 or len(self.val_tokens) == 0 or len(self.test_tokens) == 0:
            logger.warning("One or more data splits are empty. Check dataset size and split ratios.")
            
    def encode_string(self, s: str): # Renamed from 'encoding' for clarity
        """Encodes a string to a list of token IDs."""
        return self.tokenizer.encode(s)
    
    def decode_ids(self, ids: list): # Renamed from 'decoding' for clarity
        """Decodes a list of token IDs back to a string."""
        return self.tokenizer.decode(ids)

    def get_batch(self, split='train'):
        """
        Generates a batch of input sequences (x) and target sequences (y)
        from the specified tokenized data split (train, validation, or test).
        """
        if split == 'train':
            data_tokens = self.train_tokens
        elif split == 'validation': # Changed from 'validation' to 'val' to match attribute name
            data_tokens = self.val_tokens
        elif split == 'test':
            data_tokens = self.test_tokens
        else:
            raise ValueError(f"Unknown split type: {split}")

        # Ensure there's enough data to form a complete sequence of `time_steps + 1`.
        max_start_idx = len(data_tokens) - self.time_steps - 1
        if max_start_idx < 0:
             raise ValueError(
                 f"Dataset split '{split}' is too small ({len(data_tokens)} tokens) "
                 f"for time_steps={self.time_steps}. Need at least {self.time_steps + 1} tokens. "
                 f"Consider a smaller time_steps, larger dataset, or different split ratios."
            )
        
        # Randomly select `batch_size` starting indices.
        # Ensure randint upper bound is exclusive, so max_start_idx + 1
        ix = torch.randint(0, max_start_idx + 1, (self.batch_size,))

        # Create input sequences (x) and target sequences (y) from token IDs.
        # We are creating batches such as this because we are doing self-attention training with masking.
        # Example:
        # If x sequence is: ['twinkle', 'twinkle', 'little', 'star']
        # The model's task for each position (due to causal masking) is:
        # - given 'twinkle', predict 'twinkle'
        # - given 'twinkle twinkle', predict 'little'
        # - given 'twinkle twinkle little', predict 'star'
        # - given 'twinkle twinkle little star', predict ',' (or whatever the next token in the full text is)
        x = torch.stack([data_tokens[i : i + self.time_steps] for i in ix])
        y = torch.stack([data_tokens[i + 1 : i + self.time_steps + 1] for i in ix])
        
        # Note: Moving to device will be handled in the training loop / evaluation function
        return x, y
    
class TransformerBlock(nn.Module):
    def __init__(self, channel_dim, num_heads, context_window):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim) # input to attention heads is normalised
        self.attn = MultiHeadAttention(channel_dim, num_heads, context_window)
        self.ln2 = nn.LayerNorm(channel_dim) # input to feed forward network is normalised 
        self.ffn = nn.Sequential(
            nn.Linear(channel_dim, 4 * channel_dim),
            nn.ReLU(),  # or GELU for GPT-style
            nn.Linear(4 * channel_dim, channel_dim)
        )
        # This FFN expands the token's features to a higher dimension, applies non-linearity, and then projects them back to the original dimension to enhance representation.


    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # This line passes the data through attention, then adds the result back to the original input (x).
        # This 'skip connection' helps the model train deeper by allowing information and gradients to flow more easily.
        
        # Feedforward with residual
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, channel_dim, context_window, num_heads=8, num_layers=6):
        super().__init__()

        # Embedding table: maps token indices to vectors of size channel_dim
        self.token_embedding = nn.Embedding(vocab_size, channel_dim)

        # Positional embedding: one vector per position up to context_window
        self.position_embedding = nn.Embedding(context_window, channel_dim)
        
        # Stack of Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channel_dim, num_heads, context_window)
            for _ in range(num_layers)
        ])

        # Final normalization and output projection
        self.ln_f = nn.LayerNorm(channel_dim)
        self.proj = nn.Linear(channel_dim, vocab_size)

        self.context_window = context_window

    def forward(self, x_indices, targets=None):
        # x: (B, T) where T = context length (sequence of token indices)

        B, T = x_indices.shape
        
        if T > self.context_window:
             logger.warning(f"Input sequence length {T} exceeds model context window {self.context_window}. Truncating.")
             x_indices = x_indices[:, :self.context_window] # Simple truncation
             T = self.context_window
             if targets is not None:
                 targets = targets[:, :self.context_window]

        # Step 1: Embed tokens → (B, T, C)
        token_emb = self.token_embedding(x_indices)

        # Step 2: Embed positions (broadcasts across batch)
        pos_indices = torch.arange(T, device=x_indices.device)
        pos_emb = self.position_embedding(pos_indices)  # (T, C)

        # Step 3: Combine token + position embeddings
        x = token_emb + pos_emb  # (B, T, C)
        
        for block in self.transformer_blocks:
            x = block(x)  

        x = self.ln_f(x)                                         # Final layer norm
        logits = self.proj(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Ensure targets are also on the same device as logits
            targets = targets.to(logits.device)
            B_logits, T_logits, C_logits = logits.shape # C_logits is vocab_size
            logits_for_loss = logits.view(B_logits * T_logits, C_logits)
            targets_for_loss = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
            
        return logits, loss
    
    @torch.no_grad()
    def evaluate_validation_loss(self, prep_obj, split = 'validation', eval_iters=20):
        """
        Computes average validation loss over `eval_iters` mini-batches.
        """
        self.eval()
        losses = []
        for _ in range(eval_iters):
            try:
                xb, yb = prep_obj.get_batch(split)
                xb, yb = xb.to(DEVICE), yb.to(DEVICE) # Move batch to DEVICE
                _, loss = self(xb, yb)
                losses.append(loss.item())
            except ValueError as e: # Catch errors if a split is too small for a batch
                logger.warning(f"Skipping a batch for split '{split}' during evaluation due to: {e}")
                continue 
        self.train()
        
        return sum(losses) / len(losses) if losses else float('nan')
    
    def train_loop(self, prep_obj, steps=2000, val_check_every=50, patience=4, min_delta=1e-4, lr=5e-3):
        """
        Training loop with early stopping, validation checks, and plotting.
        """
        self.to(DEVICE) # Move model to device
        logger.info(f"Starting training on {DEVICE}")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        stale_checks = 0
        checkpoint_path = "transformer_checkpoint_wordseq.pt"

        train_losses = []
        val_loss_dict = {}
        self.train()

        for step in range(1, steps + 1):
            xb, yb = prep_obj.get_batch('train')
            xb, yb = xb.to(DEVICE), yb.to(DEVICE) # Move batch to device
            logits, loss = self(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if step % val_check_every == 0 or step == steps: # Also check at the very last step
                val_loss = self.evaluate_validation_loss(prep_obj,split = 'validation', eval_iters=20) # Pass prep_obj
                val_loss_dict[step] = val_loss

                logger.info(f"[Step {step}/{steps}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
                print(f"[Step {step}/{steps}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")


                if val_loss < best_val_loss - min_delta : # Check if val_loss is better
                    best_val_loss = val_loss
                    stale_checks = 0
                    torch.save(self.state_dict(), checkpoint_path)
                    logger.info(f" ↑ New Best Model (Val Loss: {best_val_loss:.4f}); Checkpoint saved to {checkpoint_path}")
                    print(" ↑ New Best Model; Checkpoint saved.")
                else:
                    stale_checks += 1
                    logger.info(f" No improvement for {stale_checks}/{patience} validation checks.")
                    print(f" No improvement for {stale_checks}/{patience} validation checks.")

                if stale_checks >= patience:
                    logger.warning(">>> Early stopping triggered.")
                    print(">>> Early stopping triggered.")
                    break
        
        if train_losses and val_loss_dict: # Plotting
            plt.figure(figsize=(12, 7))
            plt.plot(train_losses, label='Training Loss (per step)', color='lightblue', alpha=0.7, linewidth=1)
            
            if len(train_losses) >= val_check_every: # Smoothed training loss
                smoothed_train_loss = [sum(train_losses[i:i+val_check_every])/val_check_every for i in range(0, len(train_losses) - val_check_every + 1, val_check_every)]
                steps_for_smoothed = list(range(val_check_every, len(train_losses)+1, val_check_every))[:len(smoothed_train_loss)]
                plt.plot(steps_for_smoothed, smoothed_train_loss, label=f'Smoothed Training Loss (avg over {val_check_every} steps)', color='blue', linewidth=2)

            plt.plot(list(val_loss_dict.keys()), list(val_loss_dict.values()), 'o-', label='Validation Loss', color='orange', linewidth=2, markersize=5)
            
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('loss_curve_modified.png')
            logger.info("Loss curve saved to loss_curve_modified.png")
            # plt.show() # plt.show() can block execution in some environments, make it optional
        else:
            logger.info("No data for plotting loss curves.")


        return train_losses, val_loss_dict
    
    @torch.no_grad() # Added decorator
    def generate(self, input_ids, max_tokens_ahead = 100, temperature=1.0, top_k=None): # Added temp and top_k
        """
        Autoregressively generate new tokens, starting from input_ids.
        input_ids: tensor of shape (B, T') where T' ≤ context_window
        Returns: tensor of shape (B, T' + max_tokens_ahead)
        """
        self.eval() # Set to eval mode
        # device = next(self.parameters()).device # Get model's device
        input_ids = input_ids.to(DEVICE) # Ensure input_ids are on the correct device

        for _ in range(max_tokens_ahead):
            # Truncate to the last `context_window` tokens if too long
            if input_ids.size(1) > self.context_window:
                input_condensed = input_ids[:, -self.context_window:]
            else:
                input_condensed = input_ids
            
            logits, _ = self.forward(input_condensed) # self() calls forward
            
            last_logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature > 0: # temperature=0 can lead to issues with multinomial if not handled
                 last_logits = last_logits / temperature
            # Temperature > 1 (e.g., 2.0): Dividing the logits by a value greater than 1 makes the logits smaller (closer to zero). When these smaller logits are passed through the softmax function, the resulting probability distribution becomes smoother and more uniform. This means more tokens will have similar (non-zero) probabilities, leading to more diverse and potentially more "creative" or "random" output.
            # Temperature = 1.0: This is the default or "no change" setting. The logits are used as they are.
            # Temperature < 1 (e.g., 0.5): Dividing the logits by a value less than 1 (e.g., 0.5 is equivalent to multiplying by 2) makes the logits larger (more extreme). When these more extreme logits are passed through softmax, the probability distribution becomes sharper, concentrating most of the probability mass on the few most likely tokens. This leads to less diverse and more "deterministic" or "conservative" output.
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1))) # identifies the top_k highest logit values (v) and their indices for each sequence in the batch. min(top_k, last_logits.size(-1)) ensures you don't ask for more k values than are available in the vocabulary.
                last_logits[last_logits < v[:, [-1]]] = -float('Inf') # v[:, [-1]]: This specifically extracts the k-th highest logit value.
                # Any logit value that is smaller than the k-th highest logit is set to negative infinity (-float('Inf')).
            
            probs = F.softmax(last_logits, dim=-1) # When softmax is applied, -float('Inf') becomes 0 probability.
            
            # top_k effectively zeroes out the probabilities of all tokens except for the k most probable ones. This restricts the sampling pool to only the most plausible next tokens.
            
            # Sample next token
            # Avoid issues with temperature=0 if it results in all zeros after softmax (unlikely with top_k)
            if temperature == 0: # Deterministic: take the most probable token
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                # Temperature = 0: This is a special case handled by your code (if temperature == 0: next_token = torch.argmax(probs, dim=-1, keepdim=True)). It effectively means deterministic sampling, where the model always picks the token with the absolute highest logit (and thus highest probability after softmax).
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

# === Hyperparameter Search (Modified to accept tokenizer) ===
def hyperparameter_search(raw_text, tokenizer, lrs=[1e-2], batch_sizes=[4], time_steps_list=[8]): # Renamed time_steps to time_steps_list
    results = []
    base_model_params = {'num_heads': 8, 'num_layers': 6, 'channel_dim': 64} # Example base params

    for bs in batch_sizes:
        for ts in time_steps_list: # Use new name
            for lr in lrs:
                try:
                    print(f"Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")
                    logger.info(f"Starting Tuning run: Batch Size: {bs}, Context Window: {ts}, Learning Rate: {lr}")

                    # KEY CHANGE: Pass tokenizer to PreprocessingTraining
                    prep = PreprocessingTraining(raw_text, tokenizer=tokenizer, batch_size=bs, time_steps=ts)
                    
                    # Ensure vocab_size and context_window are correctly passed from prep object
                    model = TransformerModel(
                        vocab_size=prep.vocab_size, 
                        context_window=prep.time_steps, # Use prep.time_steps
                        channel_dim=base_model_params['channel_dim'], # Use defined channel_dim
                        num_heads=base_model_params['num_heads'],
                        num_layers=base_model_params['num_layers']
                    ).to(DEVICE) # Move model to device

                    # Pass prep object to train_loop and evaluate_validation_loss
                    model.train_loop(prep, steps=1000, val_check_every=20, patience=6, lr=lr) # Reduced steps for tuning

                    val_loss = model.evaluate_validation_loss(prep, split = 'validation') # Pass prep object
                    results.append({
                        'batch_size': bs,
                        'time_steps': ts,
                        'learning_rate': lr,
                        'val_loss': val_loss
                    })
                    logger.info(f"Completed run: bs={bs}, ts={ts}, lr={lr}, val_loss={val_loss:.4f}")
                except Exception as e:
                    logger.error(f"Error in tuning run (bs={bs}, ts={ts}, lr={lr}): {e}", exc_info=True)
    return results

if __name__ == '__main__':
    try:
        input_file_path = 'input.txt' # For TinyShakespeare dataset
        if not os.path.exists(input_file_path):
            logger.critical(f"{input_file_path} (for TinyShakespeare) not found. Please ensure the dataset file exists.")
            print(f"CRITICAL ERROR: {input_file_path} (for TinyShakespeare) not found. Please place your dataset in this file.")
            exit()

        with open(input_file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        logger.info(f"Successfully loaded {input_file_path}")

        torch.manual_seed(1337)

        # --- Tokenizer Initialization ---
        tokenizer_name = "gpt2"
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token 
                logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
        except Exception as e:
            logger.critical(f"Could not load tokenizer '{tokenizer_name}'. Error: {e}", exc_info=True)
            print(f"Error: Could not load tokenizer '{tokenizer_name}'. Ensure internet access or cached tokenizer.")
            exit()
        
        # --- Hyperparameters ---
        batch_s = 16      # Batch size
        time_s = 64       # Context window (sequence length)
        channel_d = 128   # Embedding dimension
        num_h = 8         # Number of attention heads
        num_l = 6         # Number of transformer layers
        learning_r = 5e-4 # Learning rate (adjusted from 5e-3, often 3e-4 to 5e-4 is good for Adam)
        training_steps = 2000
        val_check = 100    # Validate every N steps
        train_patience = 5 # Early stopping patience

        # --- Preprocessing ---
        prep = PreprocessingTraining(raw_text, tokenizer = tokenizer, batch_size=batch_s, time_steps=time_s)
        
        # --- Model Initialization ---
        model = TransformerModel(
            vocab_size=prep.vocab_size, 
            channel_dim=channel_d, 
            context_window=prep.time_steps, # Use time_steps from prep object
            num_heads=num_h, 
            num_layers=num_l
        ).to(DEVICE)
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters on {DEVICE}.")

        # --- Initial Loss Check ---
        try:
            xb, yb = prep.get_batch("train")
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            expected_initial_loss = torch.log(torch.tensor(prep.vocab_size, dtype=torch.float)).item()
            _, loss_before = model(xb, yb)
            logger.info(f"Initial loss before training: {loss_before.item():.4f} (Expected for random ~{expected_initial_loss:.2f})")
            print(f"Initial loss before training: {loss_before.item():.4f} (Expected for random ~{expected_initial_loss:.2f})")
        except Exception as e:
            logger.error(f"Could not perform initial loss check: {e}", exc_info=True)


        # --- Training ---
        should_train = True
        if should_train:
            logger.info("Starting training...")
            model.train_loop(prep, steps=training_steps, val_check_every=val_check, patience=train_patience, lr=learning_r)
            logger.info("Training finished.")

            try: # Load the best model saved during training for subsequent evaluations
                model.load_state_dict(torch.load("transformer_checkpoint_wordseq.pt", map_location=DEVICE))
                logger.info("Loaded best model from checkpoint for final evaluations.")
            except FileNotFoundError:
                logger.warning("Checkpoint file not found after training. Using current model state for evaluations.")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}", exc_info=True)

        # --- Post-Training Evaluation on a sample training batch ---
        if 'xb' in locals() and 'yb' in locals(): # Check if xb, yb were successfully created
            _, loss_after = model(xb, yb) 
            logger.info(f"Post-training loss (on one sample train batch): {loss_after.item():.4f}")
            print(f"Post-training loss (on one sample train batch): {loss_after.item():.4f}")

        # --- Evaluate on Test Set ---
        logger.info("Evaluating on test set...")
        test_loss_avg = model.evaluate_validation_loss(prep, split='test', eval_iters=50) # Use more iters for test
        # Evaluate the value and format it conditionally
        if test_loss_avg is not None and not torch.isnan(torch.tensor(test_loss_avg)):
            formatted_test_loss = f"{test_loss_avg:.4f}"
        else:
            formatted_test_loss = 'N/A'

        logger.info(f"Average Test Set Loss: {formatted_test_loss}")
        print(f"Average Test Set Loss: {formatted_test_loss}")

        # --- Text Generation ---
        logger.info("Generating text...")
        prompt_text = 'Romeo, Romeo, wherefore art thou' # A classic prompt
        input_ids = torch.tensor([prep.encode_string(prompt_text)], dtype=torch.long).to(DEVICE)
        
        generated_ids = model.generate(input_ids, max_tokens_ahead=50, temperature=0.7, top_k=40)
        
        generated_text = prep.decode_ids(generated_ids[0].cpu().tolist())
        print(f"\n--- Generated Text (Prompt: '{prompt_text}') ---")
        print(generated_text)
        print("--- End of Generated Text ---")
        logger.info(f"Generated text for prompt '{prompt_text}': {generated_text}")

    except FileNotFoundError: # This specific catch might be redundant if path check at start exits
        logger.critical(f"{input_file_path} (for TinyShakespeare) not found. Please ensure the dataset file exists.", exc_info=True)
        print(f"CRITICAL ERROR: {input_file_path} (for TinyShakespeare) not found. Please ensure the dataset file exists in the correct location.")
    except ValueError as e:
        logger.critical(f"Value error during execution: {e}", exc_info=True)
        print(f"Value error: {e}")
    except KeyError as e:
        logger.critical(f"Key error encountered: {e}", exc_info=True)
        print(f"Key error: {e}")
    except RuntimeError as e:
        logger.critical(f"Runtime error: {e}", exc_info=True)
        print(f"Runtime error: {e}")
        if "CUDA out of memory" in str(e):
            print("Hint: CUDA out of memory. Try reducing batch_size, context_window, or model size (channel_dim, num_layers).")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in main execution: {e}", exc_info=True)
        print(f"An unhandled error occurred: {e}")
        
    # == Optional: Hyperparameter Search Execution ==
    try:
        should_tune = False # Set to True to run hyperparameter search
        if should_tune:
            logger.info("Starting hyperparameter search...")
            if 'raw_text' not in locals() or 'tokenizer' not in locals():
                logger.error("raw_text or tokenizer not defined for hyperparameter search. Skipping.")
                print("Error: raw_text or tokenizer not available for hyperparameter search.")
            else:
                lrs_tune = [1e-3, 5e-4]
                batch_sizes_tune = [8, 16]
                time_steps_param_tune = [32, 64]
                
                tuning_results = hyperparameter_search(
                    raw_text, 
                    tokenizer, 
                    lrs=lrs_tune, 
                    batch_sizes=batch_sizes_tune, 
                    time_steps_list=time_steps_param_tune
                )

                if tuning_results:
                    # Filter out None val_loss results before sorting
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
    except Exception as e:
        logger.critical(f"Unhandled error during hyperparameter tuning: {e}", exc_info=True)
        print(f"Error during hyperparameter tuning: {e}")

        