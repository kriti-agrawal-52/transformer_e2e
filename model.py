import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__) # Use __name__ for module-specific logging

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, context_window, dropout_rate=0.2):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim, bias = False)
        self.key = nn.Linear(embed_dim, embed_dim, bias = False)
        self.value = nn.Linear(embed_dim, embed_dim, bias = False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))
        self.attn_dropout = nn.Dropout(dropout_rate) # Dropout after softmax attention weights
        self.proj_dropout = nn.Dropout(dropout_rate) # Dropout after output projection

    def forward(self, x):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(B, T, H, D).transpose(1, 2)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)
        attn_scores = Q @ K.transpose(-2, -1)
        attn_scores = attn_scores / (D ** 0.5)
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights) # Apply dropout here
        attn_output = attn_weights @ V
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_dropout(self.out_proj(out)) # Apply dropout here

class TransformerBlock(nn.Module):
    def __init__(self, channel_dim, num_heads, context_window, dropout_rate=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.attn = MultiHeadAttention(channel_dim, num_heads, context_window, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate) # Dropout after attention (before residual)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.ffn = nn.Sequential(
            nn.Linear(channel_dim, 4 * channel_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout within FFN
            nn.Linear(4 * channel_dim, channel_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate) # Dropout after FFN (before residual)

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.ln1(x))) 
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, channel_dim, context_window, num_heads=8, num_layers=6, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, channel_dim)
        self.position_embedding = nn.Embedding(context_window, channel_dim) 
        self.pos_emb_dropout = nn.Dropout(dropout_rate) # Dropout after position embeddings
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channel_dim, num_heads, context_window, dropout_rate) # Pass dropout_rate to blocks
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(channel_dim)
        self.proj = nn.Linear(channel_dim, vocab_size)
        self.context_window = context_window 

    def forward(self, x_indices, targets=None):
        B, T = x_indices.shape
        
        # This internal truncation logic is fine for training, but generate() handles it for inference
        if T > self.context_window:
             logger.warning(f"Input sequence length {T} exceeds model context window {self.context_window}. Truncating.") 
             x_indices = x_indices[:, :self.context_window] 
             T = self.context_window 
             if targets is not None:
                 targets = targets[:, :self.context_window] 

        token_emb = self.token_embedding(x_indices) 
        pos_indices = torch.arange(T, device=x_indices.device) 
        pos_emb = self.position_embedding(pos_indices) 
        x = token_emb + pos_emb 
        x = self.pos_emb_dropout(x) # Apply dropout after summing embeddings
        
        for block in self.transformer_blocks: 
            x = block(x)  
        x = self.ln_f(x) 
        logits = self.proj(x) 
        
        loss = None
        if targets is not None: 
            targets = targets.to(logits.device) 
            B_logits, T_logits, C_logits = logits.shape 
            logits_for_loss = logits.view(B_logits * T_logits, C_logits) 
            targets_for_loss = targets.view(B_logits * T_logits) 
            loss = F.cross_entropy(logits_for_loss, targets_for_loss) 
            
        return logits, loss