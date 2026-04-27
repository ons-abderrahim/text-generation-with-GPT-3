"""
generate.py
-----------
Text generation with multiple decoding strategies:
  - Greedy decoding
  - Beam search
  - Top-K sampling
  - Nucleus (Top-P) sampling
  - Temperature scaling
  - Contrastive search
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation config dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """All parameters controlling text generation."""

    # Length
    max_new_tokens: int = 200
    min_new_tokens: int = 10

    # Strategy
    strategy: str = "nucleus"           # greedy | beam | topk | nucleus | contrastive
    num_beams: int = 5                  # For beam search
    do_sample: bool = True              # For sampling strategies
    temperature: float = 0.8            # Scales logit distribution
    top_k: int = 50                     # Top-K sampling
    top_p: float = 0.92                 # Nucleus sampling (top-p)
    typical_p: float = 1.0              # Typical decoding

    # Repetition
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3

    # Contrastive search
    penalty_alpha: float = 0.6

    # Output
    num_return_sequences: int = 1
    skip_special_tokens: bool = True

    def to_hf_kwargs(self) -> dict:
        """Convert to kwargs for model.generate()."""
        base = dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_return_sequences=self.num_return_sequences,
        )
        if self.strategy == "greedy":
            base.update(do_sample=False, num_beams=1)
        elif self.strategy == "beam":
            base.update(do_sample=False, num_beams=self.num_beams)
        elif self.strategy == "topk":
            base.update(do_sample=True, top_k=self.top_k, temperature=self.temperature)
        elif self.strategy == "nucleus":
            base.update(do_sample=True, top_p=self.top_p, top_k=0, temperature=self.temperature)
        elif self.strategy == "contrastive":
            base.update(
                do_sample=False,
                penalty_alpha=self.penalty_alpha,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. "
                             "Choose from: greedy, beam, topk, nucleus, contrastive")
        return base


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Union[str, List[str]],
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Generate text from one or more prompts.

    Args:
        model:     Fine-tuned causal language model.
        tokenizer: Corresponding tokenizer.
        prompts:   A single prompt string or a list of prompts.
        config:    GenerationConfig (uses defaults if None).
        device:    Target device; auto-detected if None.

    Returns:
        List of generated strings (one per prompt × num_return_sequences).
    """
    if config is None:
        config = GenerationConfig()

    if isinstance(prompts, str):
        prompts = [prompts]

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_outputs: List[str] = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            gen_kwargs = config.to_hf_kwargs()
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

            output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode only the newly generated tokens
            for seq in output_ids:
                new_tokens = seq[prompt_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=config.skip_special_tokens)
                all_outputs.append(text.strip())

    return all_outputs


# ---------------------------------------------------------------------------
# Interactive generator class
# ---------------------------------------------------------------------------

class TextGenerator:
    """
    Stateful text generator for repeated inference.

    Example:
        generator = TextGenerator.from_pretrained("outputs/checkpoints/gpt2-ag-news")
        texts = generator.generate("Breaking news:", temperature=0.7)
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(cls, model_path: str, **model_kwargs) -> "TextGenerator":
        """Load model + tokenizer from a saved checkpoint."""
        from src.model import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path, **model_kwargs)
        return cls(model, tokenizer)

    def generate(
        self,
        prompt: str,
        strategy: str = "nucleus",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.92,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> List[str]:
        """Generate text with a simple interface."""
        config = GenerationConfig(
            strategy=strategy,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            **{k: v for k, v in kwargs.items() if hasattr(GenerationConfig, k)},
        )
        return generate_text(self.model, self.tokenizer, prompt, config, self.device)

    def stream(self, prompt: str, max_new_tokens: int = 200, **kwargs):
        """
        Token-by-token streaming generator (requires transformers ≥ 4.38).

        Yields decoded tokens one at a time for real-time display.
        """
        from transformers import TextStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=kwargs.get("temperature", 0.8),
            top_p=kwargs.get("top_p", 0.92),
            pad_token_id=self.tokenizer.eos_token_id,
        )
