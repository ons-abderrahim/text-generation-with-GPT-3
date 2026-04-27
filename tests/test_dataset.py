"""
Tests for src/dataset.py
"""

import pytest
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def tiny_dataset():
    """A minimal in-memory DatasetDict for testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Scientists discovered a new species in the Amazon rainforest.",
        "Stock markets surged today amid positive economic data.",
        "The government announced new climate change policies yesterday.",
        "A new study shows coffee may improve cognitive function.",
    ] * 20  # Repeat to ensure we have enough for blocking

    train_data = Dataset.from_dict({"text": texts[:80]})
    val_data = Dataset.from_dict({"text": texts[80:]})
    return DatasetDict({"train": train_data, "validation": val_data})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadTextDataset:
    def test_loads_huggingface_dataset(self):
        from src.dataset import load_text_dataset
        ds = load_text_dataset("ag_news", train_split="train", eval_split="test")
        assert isinstance(ds, DatasetDict)
        assert "train" in ds
        assert "validation" in ds
        assert len(ds["train"]) > 0

    def test_creates_validation_split_if_missing(self, tmp_path):
        from src.dataset import load_text_dataset
        # ag_news only has train/test — should auto-create validation
        ds = load_text_dataset("ag_news", train_split="train", eval_split="test")
        assert "validation" in ds


class TestTokenizeDataset:
    def test_produces_correct_keys(self, tiny_dataset, tokenizer):
        from src.dataset import tokenize_dataset
        lm_ds = tokenize_dataset(tiny_dataset, tokenizer, max_length=64)
        assert "input_ids" in lm_ds["train"].column_names
        assert "labels" in lm_ds["train"].column_names

    def test_block_length_is_correct(self, tiny_dataset, tokenizer):
        from src.dataset import tokenize_dataset
        max_length = 64
        lm_ds = tokenize_dataset(tiny_dataset, tokenizer, max_length=max_length)
        for example in lm_ds["train"]:
            assert len(example["input_ids"]) == max_length

    def test_labels_equal_input_ids(self, tiny_dataset, tokenizer):
        from src.dataset import tokenize_dataset
        lm_ds = tokenize_dataset(tiny_dataset, tokenizer, max_length=64)
        for example in lm_ds["train"]:
            assert example["input_ids"] == example["labels"]

    def test_validation_split_preserved(self, tiny_dataset, tokenizer):
        from src.dataset import tokenize_dataset
        lm_ds = tokenize_dataset(tiny_dataset, tokenizer, max_length=64)
        assert "validation" in lm_ds
        assert len(lm_ds["validation"]) > 0


class TestGetDataloaders:
    def test_returns_two_dataloaders(self, tiny_dataset, tokenizer):
        from src.dataset import tokenize_dataset, get_dataloaders
        lm_ds = tokenize_dataset(tiny_dataset, tokenizer, max_length=64)
        train_loader, eval_loader = get_dataloaders(lm_ds, train_batch_size=2, eval_batch_size=2)
        assert train_loader is not None
        assert eval_loader is not None

    def test_batch_shape(self, tiny_dataset, tokenizer):
        from src.dataset import tokenize_dataset, get_dataloaders
        lm_ds = tokenize_dataset(tiny_dataset, tokenizer, max_length=64)
        train_loader, _ = get_dataloaders(lm_ds, train_batch_size=2)
        batch = next(iter(train_loader))
        assert batch["input_ids"].shape == (2, 64)
        assert batch["labels"].shape == (2, 64)
