"""
trainer.py
----------
Fine-tuning wrapper built on top of HuggingFace Trainer.
Adds custom callbacks for logging, early stopping, and W&B integration.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from datasets import DatasetDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training arguments factory
# ---------------------------------------------------------------------------

def build_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments from a config dict.

    Args:
        config: Dict matching the 'training' section of the YAML config.

    Returns:
        TrainingArguments instance.
    """
    training_cfg = config.get("training", {})
    return TrainingArguments(
        output_dir=training_cfg.get("output_dir", "outputs/checkpoints/model"),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_steps=training_cfg.get("warmup_steps", 500),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        fp16=training_cfg.get("fp16", torch.cuda.is_available()),
        bf16=training_cfg.get("bf16", False),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=training_cfg.get("logging_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        report_to=training_cfg.get("report_to", "tensorboard"),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        optim=training_cfg.get("optim", "adamw_torch"),
        seed=training_cfg.get("seed", 42),
        push_to_hub=training_cfg.get("push_to_hub", False),
        hub_model_id=training_cfg.get("hub_model_id", None),
    )


# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------

class PerplexityLoggingCallback(TrainerCallback):
    """Log perplexity (exp(eval_loss)) at each evaluation step."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            import math
            ppl = math.exp(metrics["eval_loss"])
            logger.info(f"  Step {state.global_step} | Perplexity: {ppl:.2f}")
            if state.log_history:
                state.log_history[-1]["eval_perplexity"] = ppl


class SampleGenerationCallback(TrainerCallback):
    """Generate sample text after each epoch for qualitative monitoring."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        every_n_epochs: int = 1,
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts or ["Once upon a time,", "The latest news:"]
        self.max_new_tokens = max_new_tokens
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.epoch % self.every_n_epochs != 0 or model is None:
            return
        model.eval()
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample generations — Epoch {int(state.epoch)}")
        logger.info("=" * 60)
        with torch.no_grad():
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.92,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                logger.info(f"\nPrompt: '{prompt}'\n{generated}\n")
        model.train()


# ---------------------------------------------------------------------------
# Fine-tuning entry point
# ---------------------------------------------------------------------------

def fine_tune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    lm_dataset: DatasetDict,
    config: Dict[str, Any],
    sample_prompts: Optional[List[str]] = None,
    early_stopping_patience: int = 3,
) -> Trainer:
    """
    Fine-tune a causal language model using HuggingFace Trainer.

    Args:
        model:                    The pre-trained model to fine-tune.
        tokenizer:                The corresponding tokenizer.
        lm_dataset:               Tokenized DatasetDict with train/validation.
        config:                   Full config dict (from YAML).
        sample_prompts:           Prompts for qualitative generation during training.
        early_stopping_patience:  Stop if eval_loss doesn't improve for N evals.

    Returns:
        Trained HuggingFace Trainer instance.
    """
    training_args = build_training_args(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM — no masked language modeling
    )

    callbacks = [
        PerplexityLoggingCallback(),
        SampleGenerationCallback(tokenizer=tokenizer, prompts=sample_prompts),
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("Starting fine-tuning...")
    logger.info(f"  Output dir: {training_args.output_dir}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Train batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: "
                f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Fine-tuning complete.")
    logger.info(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    return trainer
