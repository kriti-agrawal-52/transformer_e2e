import pytest
import torch
from transformers import AutoTokenizer
import os
import sys

# Add src to path to allow direct import
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from data.processing import PreprocessingTraining


@pytest.fixture(scope="module")
def tokenizer():
    """Provides a tokenizer for the tests."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.mark.unit
def test_preprocessing_initialization(dummy_data_path, tokenizer):
    """Tests if PreprocessingTraining initializes correctly."""
    with open(dummy_data_path, "r") as f:
        text = f.read()

    preprocess = PreprocessingTraining(text, tokenizer, batch_size=2, time_steps=8)

    assert preprocess.batch_size == 2
    assert preprocess.time_steps == 8
    assert len(preprocess.all_token_ids) > 0
    assert isinstance(preprocess.all_token_ids, list)
    assert preprocess.vocab_size == tokenizer.vocab_size


@pytest.mark.unit
def test_data_splitting(dummy_data_path, tokenizer):
    """Tests if data is split into train, val, and test sets correctly."""
    with open(dummy_data_path, "r") as f:
        text = f.read()

    preprocess = PreprocessingTraining(text, tokenizer, batch_size=2, time_steps=8)

    train_len = len(preprocess.train_data)
    val_len = len(preprocess.val_data)
    test_len = len(preprocess.test_data)

    assert train_len > 0
    assert val_len > 0
    assert test_len > 0
    assert train_len + val_len + test_len == len(preprocess.all_token_ids)

    # Check if splits are roughly 80/10/10
    total_len = len(preprocess.all_token_ids)
    assert abs(train_len / total_len - 0.8) < 0.05
    assert abs(val_len / total_len - 0.1) < 0.05
    assert abs(test_len / total_len - 0.1) < 0.05


@pytest.mark.unit
def test_get_batch(dummy_data_path, tokenizer):
    """Tests if get_batch returns batches of the correct shape and type."""
    with open(dummy_data_path, "r") as f:
        text = f.read()

    preprocess = PreprocessingTraining(text, tokenizer, batch_size=4, time_steps=16)

    x, y = preprocess.get_batch("train")

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


@pytest.mark.unit
def test_empty_text_input(tokenizer):
    """Tests that providing empty text raises a ValueError."""
    with pytest.raises(ValueError, match="Input text is empty"):
        PreprocessingTraining("", tokenizer, batch_size=2, time_steps=8)


@pytest.mark.unit
def test_dataset_too_small(tokenizer):
    """Tests that a dataset smaller than a single batch raises a ValueError."""
    small_text = "This is a very small text."
    with pytest.raises(ValueError, match="Dataset is too small"):
        PreprocessingTraining(small_text, tokenizer, batch_size=8, time_steps=32)


@pytest.mark.unit
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_get_batch_from_all_splits(dummy_data_path, tokenizer, split):
    """Parametrized test to ensure get_batch works for all data splits."""
    with open(dummy_data_path, "r") as f:
        text = f.read()

    preprocess = PreprocessingTraining(text, tokenizer, batch_size=2, time_steps=8)

    x, y = preprocess.get_batch(split)

    assert x is not None
    assert y is not None
    assert x.shape == (2, 8)
    assert y.shape == (2, 8)
