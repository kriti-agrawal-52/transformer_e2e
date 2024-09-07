import pytest
import torch
import os
from unittest.mock import patch, MagicMock

from src.serving.generator import run_generation, validate_generation_params
from src.models.transformer import TransformerModel


@pytest.fixture
def setup_mocks_for_generation(mock_generate_config):
    """Centralized mock setup for generation integration tests."""
    mock_model = MagicMock(spec=TransformerModel)

    def generate_side_effect(start_tokens, max_new_tokens, *args, **kwargs):
        return torch.arange(
            start_tokens.shape[1], start_tokens.shape[1] + max_new_tokens
        ).unsqueeze(0)

    mock_model.generate.side_effect = generate_side_effect

    mock_checkpoint = {
        "model_state_dict": {"param": torch.tensor(1.0)},
        "model_args": {
            "vocab_size": 50257,
            "channel_dim": 32,
            "context_window": 16,
            "num_heads": 4,
            "num_layers": 2,
        },
    }

    with patch("torch.load", return_value=mock_checkpoint) as mock_torch_load, patch(
        "src.models.transformer.TransformerModel", return_value=mock_model
    ) as mock_model_class, patch("wandb.Api") as mock_wandb_api, patch(
        "transformers.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer:

        mock_api_instance = mock_wandb_api.return_value
        mock_run = MagicMock()
        mock_run.config = {
            "channel_dim": 32,
            "context_window": 16,
            "num_heads": 4,
            "num_layers": 2,
        }
        mock_api_instance.run.return_value = mock_run

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.return_value = [0, 1, 2]
        mock_tokenizer_instance.decode.return_value = "Generated text."
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer.return_value = mock_tokenizer_instance

        yield {
            "torch_load": mock_torch_load,
            "model_class": mock_model_class,
            "model_instance": mock_model,
            "wandb_api": mock_wandb_api,
            "tokenizer": mock_tokenizer,
            "tokenizer_instance": mock_tokenizer_instance,
        }


@pytest.mark.integration
def test_generation_successful_run(mock_generate_config, setup_mocks_for_generation):
    """Tests a complete, successful generation run."""
    run_generation(mock_generate_config)

    setup_mocks_for_generation["torch_load"].assert_called_with(
        mock_generate_config.CHECKPOINT_PATH, map_location="cpu"
    )
    setup_mocks_for_generation["model_class"].assert_called()
    setup_mocks_for_generation["model_instance"].generate.assert_called()
    setup_mocks_for_generation["tokenizer_instance"].encode.assert_called_with(
        mock_generate_config.PROMPT
    )
    setup_mocks_for_generation["tokenizer_instance"].decode.assert_called()


@pytest.mark.integration
def test_deterministic_generation_with_seed(
    mock_generate_config, setup_mocks_for_generation
):
    """Tests that with a fixed seed, the generated output is identical."""
    torch.manual_seed(42)
    run_generation(mock_generate_config)
    first_call_args = setup_mocks_for_generation["model_instance"].generate.call_args

    torch.manual_seed(42)
    run_generation(mock_generate_config)
    second_call_args = setup_mocks_for_generation["model_instance"].generate.call_args

    assert first_call_args == second_call_args


@pytest.mark.integration
def test_prompt_truncation(mock_generate_config, setup_mocks_for_generation):
    """Tests that a long prompt is correctly truncated."""
    model_context_window = 16
    long_prompt_tokens = list(range(model_context_window + 10))
    setup_mocks_for_generation["tokenizer_instance"].encode.return_value = (
        long_prompt_tokens
    )

    run_generation(mock_generate_config)

    generated_tokens = setup_mocks_for_generation["model_instance"].generate.call_args[
        0
    ][0]

    assert generated_tokens.shape[1] == model_context_window
    assert generated_tokens[0, -1] == long_prompt_tokens[-1]


@pytest.mark.integration
def test_max_tokens_limit(mock_generate_config, setup_mocks_for_generation):
    """Tests that the number of generated tokens respects the MAX_TOKENS config."""
    mock_generate_config.MAX_TOKENS = 25

    run_generation(mock_generate_config)

    max_new_tokens_arg = setup_mocks_for_generation[
        "model_instance"
    ].generate.call_args[1]
    assert max_new_tokens_arg == mock_generate_config.MAX_TOKENS


@pytest.mark.unit
@pytest.mark.parametrize(
    "param, value, error_msg",
    [
        ("TEMPERATURE", 3.0, "must be between 0.0 and 2.0"),
        ("TOP_K", -5, "must be a non-negative integer"),
        ("MAX_TOKENS", 1000, "must be between 1 and 512"),
        ("PROMPT", "", "must be a non-empty string"),
    ],
)
def test_generation_param_validation(mock_generate_config, param, value, error_msg):
    """Tests that invalid generation parameters raise a ValueError."""
    setattr(mock_generate_config, param, value)
    with pytest.raises(ValueError, match=error_msg):
        validate_generation_params(mock_generate_config)
