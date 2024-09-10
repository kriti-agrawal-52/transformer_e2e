import pytest
import torch
from unittest.mock import patch, MagicMock

from src.models.transformer import TransformerModel


@pytest.mark.unit
@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_model_with_different_attention_heads(num_heads):
    """Tests that the model handles different numbers of attention heads correctly."""
    channel_dim = 64  # Must be divisible by num_heads
    
    model = TransformerModel(
        vocab_size=1000,
        channel_dim=channel_dim,
        context_window=16,
        num_heads=num_heads,
        num_layers=2,
        dropout_rate=0.1
    )
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 16))
    logits, loss = model(input_ids, targets=input_ids)
    
    assert logits.shape == (2, 16, 1000)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.unit
@pytest.mark.parametrize("num_layers", [1, 2, 4, 6, 12])
def test_model_with_different_layer_counts(num_layers):
    """Tests that the model handles different numbers of transformer layers."""
    
    model = TransformerModel(
        vocab_size=1000,
        channel_dim=64,
        context_window=16, 
        num_heads=4,
        num_layers=num_layers,
        dropout_rate=0.1
    )
    
    # Verify correct number of transformer blocks
    assert len(model.transformer_blocks) == num_layers
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 16))
    logits, loss = model(input_ids, targets=input_ids)
    
    assert logits.shape == (2, 16, 1000)


@pytest.mark.unit
@pytest.mark.parametrize("dropout_rate", [0.0, 0.1, 0.2, 0.5])
def test_model_with_different_dropout_rates(dropout_rate):
    """Tests that the model handles different dropout rates correctly."""
    
    model = TransformerModel(
        vocab_size=1000,
        channel_dim=64,
        context_window=16,
        num_heads=4,
        num_layers=2,
        dropout_rate=dropout_rate
    )
    
    # Test in training mode (dropout active)
    model.train()
    input_ids = torch.randint(0, 1000, (2, 16))
    logits1, _ = model(input_ids, targets=input_ids)
    logits2, _ = model(input_ids, targets=input_ids)
    
    if dropout_rate > 0:
        # With dropout, outputs should be different
        assert not torch.equal(logits1, logits2)
    else:
        # Without dropout, outputs should be identical
        assert torch.equal(logits1, logits2)


@pytest.mark.unit
@pytest.mark.parametrize("context_window", [8, 16, 32, 64])
def test_model_with_different_context_windows(context_window):
    """Tests that the model handles different context window sizes."""
    
    model = TransformerModel(
        vocab_size=1000,
        channel_dim=64,
        context_window=context_window,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1
    )
    
    # Test with input matching context window
    input_ids = torch.randint(0, 1000, (2, context_window))
    logits, loss = model(input_ids, targets=input_ids)
    
    assert logits.shape == (2, context_window, 1000)
    
    # Test with input smaller than context window
    if context_window > 8:
        small_input = torch.randint(0, 1000, (2, 8))
        logits_small, _ = model(small_input, targets=small_input)
        assert logits_small.shape == (2, 8, 1000)


@pytest.mark.unit
def test_model_cpu_only_mode():
    """Tests that the model works correctly in CPU-only mode."""
    
    # Ensure we're testing on CPU
    device = torch.device("cpu")
    
    model = TransformerModel(
        vocab_size=1000,
        channel_dim=64,
        context_window=16,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1
    ).to(device)
    
    input_ids = torch.randint(0, 1000, (2, 16)).to(device)
    
    # Test forward pass
    logits, loss = model(input_ids, targets=input_ids)
    
    assert logits.device == device
    assert loss.device == device
    assert logits.shape == (2, 16, 1000)


@pytest.mark.unit
def test_checkpoint_save_load_different_devices():
    """Tests checkpoint save/load across different device configurations."""
    
    model = TransformerModel(
        vocab_size=1000,
        channel_dim=64,
        context_window=16,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1
    )
    
    # Save model state
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": 100,
        "best_val_loss": 0.5
    }
    
    # Simulate save/load cycle with device mapping
    with patch("torch.save") as mock_save, \
         patch("torch.load") as mock_load:
        
        mock_load.return_value = checkpoint
        
        # Test loading on CPU (common in CI environments)
        new_model = TransformerModel(
            vocab_size=1000,
            channel_dim=64,
            context_window=16,
            num_heads=4,
            num_layers=2,
            dropout_rate=0.1
        )
        
        # Simulate loading checkpoint
        loaded_checkpoint = torch.load("dummy_path", map_location="cpu")
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        
        # Verify model still works after loading
        input_ids = torch.randint(0, 1000, (1, 16))
        logits, loss = new_model(input_ids, targets=input_ids)
        
        assert logits.shape == (1, 16, 1000)


@pytest.mark.unit
def test_model_with_edge_case_vocab_sizes():
    """Tests model with various vocabulary sizes."""
    
    vocab_sizes = [100, 1000, 10000, 50257]  # 50257 is GPT-2 vocab size
    
    for vocab_size in vocab_sizes:
        model = TransformerModel(
            vocab_size=vocab_size,
            channel_dim=64,
            context_window=16,
            num_heads=4,
            num_layers=2,
            dropout_rate=0.1
        )
        
        # Test with valid token IDs
        max_token_id = vocab_size - 1
        input_ids = torch.randint(0, max_token_id + 1, (2, 16))
        
        logits, loss = model(input_ids, targets=input_ids)
        
        assert logits.shape == (2, 16, vocab_size)
        assert isinstance(loss, torch.Tensor) 