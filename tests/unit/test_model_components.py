import pytest
import torch
import os

from src.models.transformer import TransformerModel, MultiHeadAttention, FeedForward, Block

@pytest.fixture
def model_inputs(mock_config):
    """Provides a sample input tensor for model component tests."""
    batch_size = mock_config.BATCH_SIZE
    context_window = mock_config.CONTEXT_WINDOW
    return torch.randint(0, 50257, (batch_size, context_window)) # vocab_size=50257 (GPT-2)

@pytest.mark.unit
def test_multi_head_attention(model_inputs, mock_config):
    """Tests the MultiHeadAttention module."""
    batch, seq_len = model_inputs.shape
    config = mock_config
    
    mha = MultiHeadAttention(
        channel_dim=config.CHANNEL_DIM,
        num_heads=config.NUM_HEADS,
        context_window=config.CONTEXT_WINDOW,
        dropout_rate=config.DROPOUT_RATE
    )
    
    # We need to pass it through an embedding layer first
    embedding = torch.nn.Embedding(50257, config.CHANNEL_DIM)
    x = embedding(model_inputs)
    
    output = mha(x)
    
    assert output.shape == (batch, seq_len, config.CHANNEL_DIM)

@pytest.mark.unit
def test_feed_forward(model_inputs, mock_config):
    """Tests the FeedForward module."""
    config = mock_config
    embedding = torch.nn.Embedding(50257, config.CHANNEL_DIM)
    x = embedding(model_inputs)
    
    ff = FeedForward(channel_dim=config.CHANNEL_DIM, dropout_rate=config.DROPOUT_RATE)
    output = ff(x)
    
    assert output.shape == x.shape

@pytest.mark.unit
def test_block(model_inputs, mock_config):
    """Tests the Block module, which combines MHA and FF."""
    config = mock_config
    embedding = torch.nn.Embedding(50257, config.CHANNEL_DIM)
    x = embedding(model_inputs)
    
    block = Block(
        channel_dim=config.CHANNEL_DIM,
        num_heads=config.NUM_HEADS,
        context_window=config.CONTEXT_WINDOW,
        dropout_rate=config.DROPOUT_RATE
    )
    output = block(x)
    
    assert output.shape == x.shape

@pytest.mark.unit
def test_transformer_model_forward_pass(model_inputs, mock_config):
    """Tests the full TransformerModel forward pass."""
    config = mock_config
    model = TransformerModel(
        vocab_size=50257,
        channel_dim=config.CHANNEL_DIM,
        context_window=config.CONTEXT_WINDOW,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    )
    
    logits, loss = model(model_inputs, targets=model_inputs)
    
    assert logits.shape == (config.BATCH_SIZE, config.CONTEXT_WINDOW, 50257)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

@pytest.mark.unit
def test_transformer_model_generation(mock_config):
    """Tests the generate method of the TransformerModel."""
    config = mock_config
    model = TransformerModel(
        vocab_size=50257,
        channel_dim=config.CHANNEL_DIM,
        context_window=config.CONTEXT_WINDOW,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
    )
    
    # Start with a single token
    start_token = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = 10
    
    generated_ids = model.generate(start_token, max_new_tokens, device='cpu')
    
    assert generated_ids.shape == (1, max_new_tokens + 1)

@pytest.mark.unit
def test_multi_head_attention_causal_mask(mock_config):
    """
    Tests that the causal mask in MultiHeadAttention prevents attention to future tokens.
    """
    batch_size = 2
    seq_len = mock_config.CONTEXT_WINDOW
    embed_dim = mock_config.CHANNEL_DIM
    
    mha = MultiHeadAttention(
        channel_dim=embed_dim,
        num_heads=mock_config.NUM_HEADS,
        context_window=seq_len,
        dropout_rate=0.0 # No dropout for this test
    )
    
    # Create two inputs. The second is identical to the first, except for the last token.
    x1 = torch.randn(batch_size, seq_len, embed_dim)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(batch_size, embed_dim) # Change the last token
    
    # Get the output for both inputs
    output1 = mha(x1)
    output2 = mha(x2)
    
    # The output for all tokens *except the last one* should be identical.
    # The change in the last token should not affect any previous token's output.
    assert torch.allclose(output1[:, :-1, :], output2[:, :-1, :], atol=1e-6)
    
    # The output for the last token should be different.
    assert not torch.allclose(output1[:, -1, :], output2[:, -1, :])

@pytest.mark.unit
def test_attention_weights_sum_to_one(mock_config):
    """
    Tests that attention weights from the softmax operation sum to 1,
    using a forward hook to inspect the intermediate tensor.
    """
    mha = MultiHeadAttention(
        channel_dim=mock_config.CHANNEL_DIM,
        num_heads=mock_config.NUM_HEADS,
        context_window=mock_config.CONTEXT_WINDOW,
    )
    
    # Variable to store the attention weights from the hook
    attention_weights_storage = []
    
    # Define the hook
    def hook(module, input, output):
        attention_weights_storage.append(output)

    # Register the hook on the softmax module within the MHA
    # The softmax is not a named submodule, so we access it directly.
    # Note: This is fragile. If the MHA implementation changes, this hook may fail.
    # The softmax is applied in F.softmax, so we need to hook the dropout layer after it.
    handle = mha.attn_dropout.register_forward_hook(hook)

    # Create a dummy input and run the forward pass
    x = torch.randn(mock_config.BATCH_SIZE, mock_config.CONTEXT_WINDOW, mock_config.CHANNEL_DIM)
    mha(x)
    
    # Remove the hook immediately after use
    handle.remove()
    
    assert len(attention_weights_storage) == 1, "Hook should have been called once"
    attention_weights = attention_weights_storage[0]
    
    # Check that the weights for each token sum to 1 (within a small tolerance)
    # The shape is (B, H, T, T)
    sums = attention_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))

@pytest.mark.unit
def test_model_cleanup_deletes_params_and_buffers(mock_config):
    """
    Tests that the cleanup method actually removes model parameters and buffers,
    verifying that memory is being freed.
    """
    model = TransformerModel(
        vocab_size=50257,
        channel_dim=mock_config.CHANNEL_DIM,
        context_window=mock_config.CONTEXT_WINDOW,
        num_heads=mock_config.NUM_HEADS,
        num_layers=mock_config.NUM_LAYERS,
    )

    # Check that parameters and a specific buffer exist before cleanup
    assert len(list(model.parameters())) > 0, "Model should have parameters before cleanup."
    # Get a reference to a specific attention block to check its buffer
    attention_block = model.transformer_blocks[0].attn
    assert hasattr(attention_block, 'tril'), "Attention block should have 'tril' buffer before cleanup."

    # Run cleanup
    model.cleanup()

    # Check that parameters and buffers are gone
    assert len(list(model.parameters())) == 0, "Model should have no parameters after cleanup."
    assert not hasattr(attention_block, 'tril'), "Attention block should not have 'tril' buffer after cleanup."

@pytest.mark.unit
def test_model_cleanup(mock_config):
    """Tests that the cleanup method removes model parameters and buffers."""
    config = mock_config
    model = TransformerModel(
        vocab_size=50257,
        channel_dim=config.CHANNEL_DIM,
        context_window=config.CONTEXT_WINDOW,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
    )
    
    # Check that parameters and buffers exist before cleanup
    assert len(list(model.parameters())) > 0
    # Check a specific buffer in a submodule
    assert hasattr(model.transformer_blocks[0].attn, 'tril')

    # Run cleanup
    model.cleanup()
    
    # Check that parameters and buffers are gone
    assert len(list(model.parameters())) == 0
    assert not hasattr(model.transformer_blocks[0].attn, 'tril')
