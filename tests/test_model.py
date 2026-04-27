"""
Tests for src/model.py
"""

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model import load_tokenizer, load_model, get_model_size_mb


class TestLoadTokenizer:
    def test_loads_gpt2_tokenizer(self):
        tok = load_tokenizer("gpt2")
        assert isinstance(tok, PreTrainedTokenizerBase)

    def test_pad_token_is_set(self):
        tok = load_tokenizer("gpt2")
        assert tok.pad_token is not None

    def test_encodes_and_decodes(self):
        tok = load_tokenizer("gpt2")
        text = "Hello, world!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert text in decoded


class TestLoadModel:
    def test_loads_gpt2(self):
        model = load_model("gpt2")
        assert isinstance(model, PreTrainedModel)

    def test_model_has_parameters(self):
        model = load_model("gpt2")
        params = list(model.parameters())
        assert len(params) > 0

    def test_model_size_is_reasonable(self):
        model = load_model("gpt2")
        size_mb = get_model_size_mb(model)
        # GPT-2 base should be ~500MB in float32
        assert 400 < size_mb < 700

    def test_model_forward_pass(self):
        import torch
        from transformers import AutoTokenizer

        model = load_model("gpt2")
        tok = load_tokenizer("gpt2")
        inputs = tok("Hello world", return_tensors="pt")

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        assert outputs.loss is not None
        assert outputs.logits.shape[-1] == tok.vocab_size
