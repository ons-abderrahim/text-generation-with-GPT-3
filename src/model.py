"""
model.py
--------
Model and tokenizer loading with support for GPT-2, GPT-Neo, and LLaMA.
Includes optional LoRA (PEFT) integration and 4-bit quantization.
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# Models that need a left padding side for generation
LEFT_PAD_MODELS = {"llama", "mistral", "falcon"}


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load and configure a tokenizer for causal language modeling.

    Ensures pad_token is set and padding side is correct for the model family.

    Args:
        model_name: HuggingFace model ID or local path.

    Returns:
        Configured tokenizer.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # GPT-2 and similar models have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  Set pad_token = eos_token")

    # LLaMA / decoder-only models need left padding during generation
    family = model_name.lower()
    if any(name in family for name in LEFT_PAD_MODELS):
        tokenizer.padding_side = "left"
        logger.info("  Set padding_side = left (for generation)")

    return tokenizer


def load_model(
    model_name: str,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    """
    Load a pre-trained causal language model.

    Supports optional:
      - 4-bit / 8-bit quantization via BitsAndBytes
      - LoRA fine-tuning via PEFT

    Args:
        model_name:    HuggingFace model ID or local path.
        use_lora:      Apply LoRA adapters (reduces trainable params).
        lora_r:        LoRA rank.
        lora_alpha:    LoRA scaling factor.
        lora_dropout:  Dropout probability for LoRA layers.
        load_in_4bit:  Load model in 4-bit (requires bitsandbytes).
        load_in_8bit:  Load model in 8-bit (requires bitsandbytes).
        device_map:    Device placement strategy ("auto", "cuda", "cpu").
        torch_dtype:   Override model dtype (e.g. torch.float16).

    Returns:
        Loaded (and optionally PEFT-wrapped) model.
    """
    logger.info(f"Loading model: {model_name}")

    # ---- Quantization config ----
    bnb_config = None
    if load_in_4bit:
        logger.info("  Using 4-bit quantization (BitsAndBytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        logger.info("  Using 8-bit quantization (BitsAndBytes)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ---- Dtype ----
    if torch_dtype is None and not (load_in_4bit or load_in_8bit):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ---- Load base model ----
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map if torch.cuda.is_available() else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params:     {total_params / 1e6:.1f}M")
    logger.info(f"  Trainable params: {trainable_params / 1e6:.1f}M")

    # ---- LoRA ----
    if use_lora:
        model = _apply_lora(model, lora_r, lora_alpha, lora_dropout)

    return model


def _apply_lora(
    model: PreTrainedModel,
    r: int,
    alpha: int,
    dropout: float,
) -> PreTrainedModel:
    """Wrap the model with LoRA adapters using PEFT."""
    try:
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError("Install peft: pip install peft")

    logger.info(f"  Applying LoRA (r={r}, alpha={alpha}, dropout={dropout})")

    # Prepare quantized models for training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_get_lora_target_modules(model),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _get_lora_target_modules(model: PreTrainedModel) -> list:
    """Automatically detect attention projection modules for LoRA."""
    # Common patterns across GPT-2, GPT-Neo, LLaMA
    candidate_names = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj"]
    found = set()
    for name, _ in model.named_modules():
        for candidate in candidate_names:
            if name.endswith(candidate):
                found.add(candidate)
    if not found:
        # Fallback: all linear layers
        found = {"q_proj", "v_proj"}
    logger.info(f"  LoRA target modules: {sorted(found)}")
    return sorted(found)


def load_model_and_tokenizer(
    model_name: str, **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Convenience wrapper that loads both model and tokenizer."""
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, **kwargs)
    return model, tokenizer


def get_model_size_mb(model: PreTrainedModel) -> float:
    """Return model size in MB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 ** 2)
