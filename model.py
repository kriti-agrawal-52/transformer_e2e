import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Self-Attention mechanism.

    This module takes an input sequence and computes attention scores. It splits
    the embedding dimension into multiple 'heads', allowing the model to jointly
    attend to information from different representation subspaces. It includes
    linear projections for queries, keys, and values, scaled dot-product
    attention with causal masking (for decoder-style models), and a final
    output projection. Causal masking ensures that a token can only attend to
    previous tokens in the sequence.
    """
    
    def __init__(self, embed_dim, num_heads, context_window, dropout_rate = 0.1):
        """Initialises the MultiHeadAttentionModule"""
        super().__init__()
        assert (embed_dim % num_heads == 0), "Embedding dimension must be divisible by number of attention heads."
        # The assert statement is used to enforce that a condition must be true at runtime. If the condition fails, Python raises an AssertionError
        
        self.num_heads = num_heads # total number of heads in a transformer block
        self.head_dim = embed_dim//num_heads # size of each individual head (D)
        
        # Learnable linear projections for queries, keys, and values
        # They all map from (embed_dim → embed_dim), internally split later into heads
        self.query = nn.Linear(embed_dim, embed_dim, bias = False)
        self.key = nn.Linear(embed_dim, embed_dim, bias = False)
        self.value = nn.Linear(embed_dim, embed_dim, bias = False)
        
        # After attention, we combine the heads and project back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_window, context_window))
        )
        """
        - Lower triangular matrix for causal masking (ensures token at position i only attends to tokens ≤ i)
        Register_buffer in nn.Module basically tells pytorch that "tril" is part of model's state, 
        but is not a trainable parameter, ie it is stored in model, but not updated during training.
        """
        
        self.attn_dropout = nn.Dropout(dropout_rate)  # Dropout after softmax attention weights
        self.resid_dropout = nn.Dropout(dropout_rate)
        print(f"attention layer, dropout rate: {dropout_rate}\n")
        
    def forward(self, x):
        """Performs forward pass for multi-head attention"""
        B, T, C = x.shape  # B: batch size, T: time_steps/context_window, C: embedding_dimension
        H = self.num_heads
        D = self.head_dim
        
        # === Step 1: Linear Projections ===
        # Output Shape: (B,T,C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # === Step 2: Reshaping into multi-heads ===
        Q = Q.view(B, T, H, D).transpose(1, 2)  # gives us Q shape (B, H, T, D)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)
        
        # === Step 3: Compute Attention scores ===
        attn_scores = Q @ K.transpose(-2, -1)
        
        # === Step 4: Scale scores by sqrt(D) for gradient stability ===
        attn_scores = attn_scores / (D**0.5)
        """
         Scale the attention scores before applying softmax
         Reason:
        - Each row of Q and K is a C-dimensional vector (e.g., C = 64)
        - The dot product between Q and K is computed by multiplying corresponding elements and summing: Q ⋅ K = q₁k₁ + q₂k₂ + ... + q_Ck_C
        - So, as C increases, we're summing over more terms → the dot product value grows larger
        - This leads to attention scores (Q @ Kᵀ) with larger magnitudes when C is high
        - If attention scores are too large (e.g., [9.2, -3.5, 0.7, -4.2]), softmax becomes very sharp: softmax([9.2, -3.5, 0.7, -4.2]) → [0.999, 0.0001, 0.0008, 0.00005]
        - This is close to a one-hot vector — the model ends up attending to just one token and ignoring others
        - Each token aggregates information from only one other token → attention is no longer "distributed"
        - This also makes gradients vanish during training, especially with stacked layers
        - To fix this, we scale the scores by sqrt(C), which reduces their magnitude: Example: [1.1, -0.4, 0.08, -0.5] → softmax becomes [0.46, 0.12, 0.25, 0.17]
        - This keeps the attention distribution diffused, allowing tokens to consider multiple others, and helps maintain stable gradients
        """
        
        # === Step 5: Apply causal mask (prevent attending to future tokens) ===
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # For all positions in attn_scores where the mask value is 0 (i.e., above the diagonal), replace them with -inf so that softmax will ignore them.
        
        # === Step 6: Softmax across last dimension to get attention weights ===
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_weights = self.attn_dropout(attn_weights)  # Applying dropout
    
        # Compute weighted sum of values 
        # Weighted sum: (B, H, T, T) @ (B, H, T, D) → (B, H, T, D)
        attn_output = attn_weights @ V
        
        
        # Combine heads
        # Reshape: (B, H, T, D) → (B, T, H*D = C)
        # attn_output first is transposed, ie dimensions at index 1 and 2 are swapped (H & T), so shape after transpose : (B, T, H, D)
        # after transpose, the tensor's memory layout might become non-contigouous, elements that are logically adjacent in the new shape might not be physically adjacent in memory
        # required because view pytorch operation requires tensor to be contiguous, otherwise we will get runtime error
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        return self.resid_dropout(self.out_proj(out))  # Apply dropout here
    
class TransformerBlock(nn.Module):
    """
    Represents a single block within the Transformer architecture.

    Each Transformer block typically consists of two main sub-layers:
    1.  A Multi-Head Self-Attention mechanism.
    2.  A position-wise Feed-Forward Network (FFN).
    Layer Normalization is applied before each sub-layer, and residual
    connections (skip connections) are used around each sub-layer. This
    structure helps in training deep a_models and stabilizing learning.
    """
    def __init__(self, channel_dim, num_heads, context_window, dropout_rate = 0.2):
        """Initializes a single Transformer block."""
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)  # input to attention heads is normalised.
        self.attn = MultiHeadAttention(
            channel_dim, num_heads, context_window, dropout_rate
        )  # we get output projection from multi-head attention block
        
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.ln2 = nn.LayerNorm(channel_dim)  # input to feed forward network is normalised
        self.ffn = nn.Sequential(
            nn.Linear(channel_dim, 4 * channel_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout with FFN
            nn.Linear(4 * channel_dim, channel_dim),
        )
        # This FFN expands the token's features to a higher dimension, applies non-linearity, and then projects them back to the original dimension to enhance representation.
        
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """Performs the forward pass through one Transformer block."""
        x = x + self.dropout1(self.attn(self.ln1(x)))
        # This line passes the normalised input through attention, then adds the result back to the original input (x).
        """
         Normalizes the input tensor across features so each token vector has zero mean and unit variance.
        Helps stabilize training and smoothens the optimization landscape.
        However, during backpropagation, it divides by the standard deviation — if variance is small,
        this can scale down gradients and contribute to vanishing gradients in deep networks.
        
        Without residuals, gradients must pass through LayerNorm, attention softmax, and nonlinear FFN layers — all of which can shrink or block gradients.
        Residual connections act like "gradient highways" that allow backpropagation to bypass risky transformations.
        """
        # Feed-forward sub-layer with residual connection and layer norm
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x
    
class TransformerModel(nn.Module):
    """
    Constructs the full decoder-only Transformer language model.

    This class assembles the various components into a complete language model:
    1.  Token Embedding: Maps input token IDs to dense vectors.
    2.  Positional Embedding: Adds information about the position of each token
        in the sequence.
    3.  Transformer Blocks: A stack of multiple `TransformerBlock` layers to
        process the sequence and build rich representations.
    4.  Final Layer Normalization: Applied before the final output layer.
    5.  Output Projection (Linear Layer): Maps the final representations back
        to the vocabulary size to produce logits for predicting the next token.

    It includes methods for the forward pass, evaluation, training, and
    text generation. The training loop incorporates checkpointing and
    early stopping.
    """
    
    def __init__(self, vocab_size, channel_dim, context_window, num_heads = 8, num_layers = 6, dropout_rate = 0.2, final_dropout_multiplier = None, max_dropout_val = 0.5):
        super().__init__()
        # Embedding table: maps token indices to vectors of size channel_dim
        # We are learning our token embeddings from scratch, ie they will be initialised with random values
        self.token_embeddings = nn.Embedding(vocab_size, channel_dim)
        
        # Positional embedding: One vector per position upto the context window
        self.position_embedding = nn.Embedding(context_window, channel_dim)
        
        # Determining dropout for positional embeddings 
        # Uses the base dropout_rate
        pos_emb_dropout_val = dropout_rate # Base rate for positional embedding
        
        self.pos_emb_dropout = nn.Dropout(pos_emb_dropout_val)  # dropout after positional embeddings
        
        # Determining dropout rates for transformer blocks
        initial_layer_dropout = dropout_rate
        if final_dropout_multiplier is not None and final_dropout_multiplier != 1 and num_layers > 1:
            # calculate final dropout rate, capped by max_dropout_val
            # allows scaling up (multiplier > 1) or scaling down (multiplier < 1)
            final_layer_dropout = min(initial_layer_dropout* final_dropout_multiplier, max_dropout_val)
            final_layer_dropout = max(0.0, final_layer_dropout)  # Ensure dropout is not negative
            
            # Linearly interpolate dropout rates from initial to deep most layer ie final layer
            layer_dropout_rates = [
                    initial_layer_dropout + (final_layer_dropout - initial_layer_dropout) * (i / (num_layers - 1))
                    for i in range(num_layers)
                ]
            logger.info(f"Linearly scaling dropout from {initial_layer_dropout:.3f} to {final_layer_dropout:.3f} over {num_layers} layers. Result: {[float(f'{dr:.3f}') for dr in layer_dropout_rates]}")
        else:
            # Use the same initial_layer_dropout for all layers
            layer_dropout_rates = [initial_layer_dropout] * num_layers
            logger.info(f"Using uniform dropout rate of {initial_layer_dropout} for all {num_layers} layers.")
        
            
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(channel_dim, num_heads, context_window, dropout_rate = layer_dropout_rates[i])  # Pass the specific dropout rate for this layer
                for i in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(channel_dim)
        self.proj = nn.Linear(channel_dim, vocab_size)
        self.context_window = context_window
    
    def forward(self, x_indices, targets = None):
        """Performs the forward pass through the entire model, calculating logits and optional loss."""
        # x: (B, T) where T = context length (sequence of token indices)
        B, T = x_indices.shape

        # Truncate input if it exceeds the context window
        if T > self.context_window:
            logger.warning(
                f"Input sequence length {T} exceeds model context window {self.context_window}. Truncating."
            )
            x_indices = x_indices[
                :, : self.context_window
            ]  # number of batches remain the same but the sequence length is truncated to context_window
            T = self.context_window
            if targets is not None:
                targets = targets[:, :self.context_window]

        # Step 1: Embed tokens → (B, T, C)
        token_emb = self.token_embeddings(x_indices)

        # Step 2: Embed positions (broadcasts across batch)
        # We basically get an embeddings for each position in a sequence, these are then applied to all sequences in all batches
        pos_indices = torch.arange(T, device=x_indices.device)  # creates a 1D tensor starting from 0 to upto T-1. x_indices tensor already exists. x_indices.device retrieves the device where the tensor stored in memory. so, the newly created tensor pos_indices is on the same device as x_indices
        pos_emb = self.position_embedding(pos_indices)  # shape (T, C)

        # Step 3
        x = token_emb + pos_emb  # Combine embeddings
        
        x = self.pos_emb_dropout(x)  # Apply dropout after summing embeddings

        # Pass the combined token and position embeddings through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization and projection to logits
        x = self.ln_f(x)
        logits = self.proj(x)  # shape: (B, T, vocab_size)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Ensure targets are also on the same device as logits
            targets = targets.to(logits.device)
            B_l, T_l, C_l = logits.shape  # C_l is vocab_size
            loss = F.cross_entropy(logits.view(B_l * T_l, C_l), targets.view(B_l * T_l))
            # we do not have to softmax on logits to get values because cross_entropy function take care of that.
            # we are reshaping logits and targets due to requirements of cross_entropy function
        return logits, loss
        
        