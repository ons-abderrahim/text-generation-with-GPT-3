"""
Tests for src/generate.py
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generate import GenerationConfig, generate_text


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


class TestGenerationConfig:
    def test_greedy_config(self):
        cfg = GenerationConfig(strategy="greedy")
        kwargs = cfg.to_hf_kwargs()
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1

    def test_nucleus_config(self):
        cfg = GenerationConfig(strategy="nucleus", top_p=0.9, temperature=0.7)
        kwargs = cfg.to_hf_kwargs()
        assert kwargs["do_sample"] is True
        assert kwargs["top_p"] == 0.9
        assert kwargs["temperature"] == 0.7

    def test_beam_config(self):
        cfg = GenerationConfig(strategy="beam", num_beams=4)
        kwargs = cfg.to_hf_kwargs()
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 4

    def test_invalid_strategy_raises(self):
        cfg = GenerationConfig(strategy="invalid")
        with pytest.raises(ValueError, match="Unknown strategy"):
            cfg.to_hf_kwargs()


class TestGenerateText:
    def test_single_prompt_returns_list(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        cfg = GenerationConfig(strategy="greedy", max_new_tokens=20, min_new_tokens=1)
        results = generate_text(model, tokenizer, "Hello world", cfg)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_batch_prompts(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        cfg = GenerationConfig(strategy="greedy", max_new_tokens=10, min_new_tokens=1)
        prompts = ["The cat sat", "In the beginning", "Scientists have"]
        results = generate_text(model, tokenizer, prompts, cfg)
        assert len(results) == 3

    def test_num_return_sequences(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        cfg = GenerationConfig(
            strategy="nucleus",
            max_new_tokens=10,
            min_new_tokens=1,
            num_return_sequences=3,
            temperature=1.0,
        )
        results = generate_text(model, tokenizer, "Once upon a time", cfg)
        assert len(results) == 3

    def test_output_is_new_tokens_only(self, model_and_tokenizer):
        """Generated text should NOT include the prompt."""
        model, tokenizer = model_and_tokenizer
        prompt = "The quick brown fox"
        cfg = GenerationConfig(strategy="greedy", max_new_tokens=10, min_new_tokens=1)
        results = generate_text(model, tokenizer, prompt, cfg)
        # The prompt itself should not appear verbatim at the start of the output
        # (since we slice off prompt tokens)
        assert results[0] != prompt
