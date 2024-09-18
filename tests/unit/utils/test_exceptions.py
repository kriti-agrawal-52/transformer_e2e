"""
Unit tests for custom exceptions.

Tests custom exception classes and their behavior.
"""

import pytest
from src.utils.exceptions import ValidationLossComputationError


class TestValidationLossComputationError:
    """Test the ValidationLossComputationError exception."""
    
    @pytest.mark.unit
    def test_exception_can_be_raised(self):
        """Test that the exception can be raised and caught."""
        with pytest.raises(ValidationLossComputationError):
            raise ValidationLossComputationError("Test error message")
    
    @pytest.mark.unit
    def test_exception_message_preserved(self):
        """Test that the exception message is preserved."""
        error_message = "Validation loss computation failed"
        
        with pytest.raises(ValidationLossComputationError) as exc_info:
            raise ValidationLossComputationError(error_message)
        
        assert str(exc_info.value) == error_message
    
    @pytest.mark.unit
    def test_exception_inheritance(self):
        """Test that the exception inherits from the correct base class."""
        error = ValidationLossComputationError("Test")
        assert isinstance(error, Exception)
        assert isinstance(error, ValidationLossComputationError) 