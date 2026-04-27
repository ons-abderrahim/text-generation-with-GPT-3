"""
dataset.py
----------
Dataset loading, tokenization, and preprocessing for causal language modeling.
Supports HuggingFace datasets and local text files.
"""

import os
import logging
from typing import Optional, Union, Dict, List
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizerBase
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main dataset loader
# ---------------------------------------------------------------------------

def load_text_dataset(
    dataset_name: str,
    text_column: str = "text",
    train_split: str = "train",
    eval_split: str = "validation",
    data_files: Optional[Union[str, List[str]]] = None,
    cache_dir: Optional[str] = None,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Load a text dataset from HuggingFace Hub or local files.

    Args:
        dataset_name:  HuggingFace dataset ID (e.g. "ag_news") or local path.
        text_column:   Column containing the raw text.
        train_split:   Name of the training split.
        eval_split:    Name of the evaluation split.
        data_files:    Paths to local files (txt / jsonl / csv).
        cache_dir:     Directory for caching downloaded data.
        num_proc:      Number of processes for parallel processing.

    Returns:
        DatasetDict with "train" and "validation" splits.
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if data_files or Path(dataset_name).exists():
        # ---- Local files ----
        extension = Path(data_files[0] if data_files else dataset_name).suffix.lstrip(".")
        extension = extension if extension in ("json", "jsonl", "csv", "txt") else "text"
        raw = load_dataset(
            "text" if extension == "txt" else extension,
            data_files=data_files or dataset_name,
            cache_dir=cache_dir,
        )
        # Rename the default column to match text_column
        if "text" not in raw["train"].column_names and text_column in raw["train"].column_names:
            pass  # Already correct
        elif "text" in raw["train"].column_names and text_column != "text":
            raw = raw.rename_column("text", text_column)
    else:
        # ---- HuggingFace Hub ----
        raw = load_dataset(dataset_name, cache_dir=cache_dir)

    # Normalise split names to "train" / "validation"
    splits: Dict[str, Dataset] = {}
    for key in raw.keys():
        if key == train_split or key == "train":
            splits["train"] = raw[key]
        elif key in (eval_split, "validation", "test"):
            splits["validation"] = raw[key]

    if "validation" not in splits and "train" in splits:
        logger.warning("No validation split found — creating one from 5% of train.")
        split = splits["train"].train_test_split(test_size=0.05, seed=42)
        splits["train"] = split["train"]
        splits["validation"] = split["test"]

    result = DatasetDict(splits)
    logger.info(f"  Train: {len(result['train']):,} examples")
    logger.info(f"  Validation: {len(result['validation']):,} examples")
    return result


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    max_length: int = 512,
    num_proc: int = 4,
    remove_columns: Optional[List[str]] = None,
) -> DatasetDict:
    """
    Tokenize and chunk a DatasetDict for causal language modeling.

    Each example is tokenized, concatenated, and split into fixed-length
    chunks of `max_length` tokens (no padding waste).

    Args:
        dataset:        Raw DatasetDict with text examples.
        tokenizer:      HuggingFace tokenizer (must have eos_token).
        text_column:    Column with raw text.
        max_length:     Token block size.
        num_proc:       Parallel workers.
        remove_columns: Columns to drop after tokenization.

    Returns:
        Tokenized DatasetDict ready for training.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cols_to_remove = remove_columns or dataset["train"].column_names

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

    def group_texts(examples):
        """Concatenate all texts and split into blocks of max_length."""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated["input_ids"])
        # Drop the remainder
        total = (total // max_length) * max_length
        result = {
            k: [v[i : i + max_length] for i in range(0, total, max_length)]
            for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=cols_to_remove,
        desc="Tokenizing",
    )

    logger.info("Grouping into fixed-length blocks...")
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping into {max_length}-token blocks",
    )

    logger.info(f"  Train blocks: {len(lm_dataset['train']):,}")
    logger.info(f"  Validation blocks: {len(lm_dataset['validation']):,}")
    return lm_dataset


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def get_dataloaders(
    lm_dataset: DatasetDict,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 2,
) -> tuple:
    """
    Return (train_loader, eval_loader) from a tokenized DatasetDict.
    """
    from transformers import DefaultDataCollator

    collator = DefaultDataCollator(return_tensors="pt")

    train_loader = DataLoader(
        lm_dataset["train"],
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        lm_dataset["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader
