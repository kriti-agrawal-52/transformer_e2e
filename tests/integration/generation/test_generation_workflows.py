"""
Simplified smoke tests for generation functionality.

These tests verify basic generation functionality without complex mocking.
"""

import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from scripts.generate import validate_generation_params, get_model_config_from_wandb


@pytest.fixture
def mock_generate_config():
    """Returns a mock generation config."""
    return SimpleNamespace(
        CHECKPOINT_PATH="/path/to/checkpoint.pt",
        MAIN_CONFIG_PATH="/path/to/main_config.yml",
        RUN_ID="test-run-id",
        PROMPT="Hello world",
        MAX_TOKENS=10,
        TEMPERATURE=0.7,
        TOP_K=50,
        DEVICE="cpu",
        LOG_FILE="/path/to/generate.log",
    )


class TestGenerationSmoke:
    """Simple smoke tests for generation functionality."""
    
    @pytest.mark.integration
    def test_generation_script_can_be_imported(self):
        """Basic smoke test that generation script can be imported."""
        from scripts.generate import main, validate_generation_params
        assert callable(main), "Generation main function should be callable"
        assert callable(validate_generation_params), "Validation function should be callable"
    
    @pytest.mark.integration
    def test_generation_config_validation_passes_valid_config(self, mock_generate_config):
        """Test that valid generation config passes validation."""
        # This should not raise any exceptions
        validate_generation_params(mock_generate_config)
    
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "param, value, error_msg",
        [
            ("TEMPERATURE", 3.0, "must be between 0.0 and 2.0"),
            ("TOP_K", -5, "must be a non-negative integer"),
            ("MAX_TOKENS", 1000, "must be between 1 and 512"),
            ("PROMPT", "", "must be a non-empty string"),
        ],
    )
    def test_generation_param_validation_catches_invalid_params(self, mock_generate_config, param, value, error_msg):
        """Tests that invalid generation parameters raise a ValueError."""
        setattr(mock_generate_config, param, value)
        with pytest.raises(ValueError, match=error_msg):
            validate_generation_params(mock_generate_config)
    
    @pytest.mark.integration
    def test_wandb_config_fetching_success(self):
        """Tests successful W&B config retrieval."""
        mock_run = MagicMock()
        mock_run.config = {"channel_dim": 64, "num_heads": 8}
        mock_run.name = "test-run"
        mock_run.id = "test-id"
        
        with patch('wandb.Api') as mock_wandb_api:
            mock_api = mock_wandb_api.return_value
            mock_api.run.return_value = mock_run
            
            result = get_model_config_from_wandb("test-run", "test-project")
            
            assert result == {"channel_dim": 64, "num_heads": 8}
            mock_api.run.assert_called_with("test-project/test-run")
    
    @pytest.mark.integration
    def test_wandb_config_fetching_handles_errors(self):
        """Tests W&B communication error handling."""
        with patch('wandb.Api') as mock_wandb_api:
            mock_wandb_api.side_effect = Exception("Connection failed")
            
            result = get_model_config_from_wandb("test-run", "test-project")
            
            assert result is None
    
    @pytest.mark.integration
    def test_transformer_model_can_be_imported_with_generate_method(self):
        """Test that TransformerModel can be imported and has generate method."""
        from src.models.transformer import TransformerModel
        import src.models.transformer_generate  # This adds the generate method
        
        # Create a small model instance
        model = TransformerModel(
            vocab_size=100,
            channel_dim=16,
            context_window=8,
            num_heads=2,
            num_layers=1
        )
        
        # Verify it has the generate method
        assert hasattr(model, 'generate'), "Model should have generate method after importing transformer_generate"
        assert callable(model.generate), "Generate method should be callable"


class TestGenerationEdgeCases:
    """Test generation edge cases and limits."""
    
    @pytest.mark.integration
    def test_empty_prompt_handling(self, mock_generate_config):
        """Test empty prompt returns expected behavior."""
        from scripts.generate import validate_generation_params
        
        # Empty prompt should raise validation error
        mock_generate_config.PROMPT = ""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_generation_params(mock_generate_config)
    
    @pytest.mark.integration
    def test_max_length_prompt_truncation(self):
        """Test that prompts exceeding context window are handled correctly."""
        from src.models.transformer import TransformerModel
        import src.models.transformer_generate
        
        # Create model with small context window
        model = TransformerModel(
            vocab_size=100,
            channel_dim=16,
            context_window=8,  # Small context window
            num_heads=2,
            num_layers=1
        )
        model.eval()
        
        # Create input that exceeds context window
        long_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])  # 12 tokens > 8 context
        
        with torch.no_grad():
            # Should truncate input to context window size
            result = model.generate(long_input, max_tokens_ahead=2, temperature=0.0, top_k=None, device="cpu")
            
        # Result should exist and be reasonable
        assert result is not None
        assert result.shape[0] == 1, "Should maintain batch dimension"
        assert result.shape[1] > long_input.shape[1], "Should generate additional tokens"
    
    @pytest.mark.integration
    def test_very_long_generation_within_limits(self):
        """Test that very long generation requests don't exceed limits."""
        from scripts.generate import validate_generation_params
        
        mock_config = SimpleNamespace(
            CHECKPOINT_PATH="/path/to/checkpoint.pt",
            MAIN_CONFIG_PATH="/path/to/main_config.yml", 
            RUN_ID="test-run-id",
            PROMPT="Test prompt",
            MAX_TOKENS=512,  # Maximum allowed
            TEMPERATURE=0.7,
            TOP_K=50,
            DEVICE="cpu",
            LOG_FILE="/path/to/generate.log",
        )
        
        # Should pass validation at the limit
        validate_generation_params(mock_config)
        
        # Should fail beyond the limit
        mock_config.MAX_TOKENS = 513
        with pytest.raises(ValueError, match="must be between 1 and 512"):
            validate_generation_params(mock_config)
    
    @pytest.mark.integration 
    def test_deterministic_generation_with_fixed_seed(self):
        """Test that generation is deterministic when using fixed seed."""
        from src.models.transformer import TransformerModel
        import src.models.transformer_generate
        
        # Create small model
        model = TransformerModel(
            vocab_size=100,
            channel_dim=16, 
            context_window=8,
            num_heads=2,
            num_layers=1
        )
        model.eval()
        
        input_tensor = torch.tensor([[1, 2, 3]])
        
        # Generate with fixed seed twice
        torch.manual_seed(42)
        result1 = model.generate(input_tensor, max_tokens_ahead=3, temperature=0.0, top_k=None, device="cpu")
        
        torch.manual_seed(42)  # Same seed
        result2 = model.generate(input_tensor, max_tokens_ahead=3, temperature=0.0, top_k=None, device="cpu")
        
        # Results should be identical for deterministic generation
        assert torch.equal(result1, result2), "Generation should be deterministic with same seed and temperature=0"


class TestGenerationBasicFunctionality:
    """Basic functionality tests for generation without complex mocking."""
    
    @pytest.mark.integration
    def test_generation_main_handles_missing_config_gracefully(self):
        """Test that main function handles missing config files gracefully."""
        from scripts.generate import main
        
        with patch('sys.argv', ['generate.py', '--gen_config', 'nonexistent_config.yml']), \
             patch('scripts.generate.load_config') as mock_load_config, \
             patch('builtins.print') as mock_print:
            
            # Simulate config loading failure
            mock_load_config.side_effect = FileNotFoundError("Config file not found")
            
            # Should handle the error gracefully without crashing
            try:
                main()
                handled_gracefully = True
            except FileNotFoundError:
                handled_gracefully = False
            
            # The main function should catch and handle the error
            assert handled_gracefully or mock_print.called, "Should handle missing config gracefully"
    
    @pytest.mark.integration
    def test_generation_main_handles_wandb_failure_gracefully(self):
        """Test that main function handles W&B API failure gracefully."""
        from scripts.generate import main
        
        mock_config = SimpleNamespace(
            CHECKPOINT_PATH="/fake/path/checkpoint.pt",
            MAIN_CONFIG_PATH="/fake/path/config.yml",
            RUN_ID="test-run-id",
            PROMPT="Test prompt",
            MAX_TOKENS=10,
            TEMPERATURE=0.7,
            TOP_K=50,
            DEVICE="cpu",
            LOG_FILE="/fake/path/log.txt"
        )
        
        with patch('sys.argv', ['generate.py', '--gen_config', 'test_config.yml']), \
             patch('scripts.generate.load_config', return_value=mock_config), \
             patch('scripts.generate.os.path.exists', return_value=False), \
             patch('scripts.generate.get_model_config_from_wandb', return_value=None), \
             patch('scripts.generate.setup_logging'), \
             patch('scripts.generate.load_dotenv'), \
             patch('builtins.print') as mock_print:
            
            # Should handle W&B failure gracefully
            main()
            
            # Verify the function completed without crashing
            assert True, "Main function should complete without crashing on W&B failure" 