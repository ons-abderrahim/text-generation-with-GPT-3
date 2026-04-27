#!/usr/bin/env python3
"""
train.py
--------
Main entry point for fine-tuning a causal language model.

Usage:
    python train.py --config configs/train_gpt2.yaml
    python train.py --config configs/train_gpt2.yaml --model_name gpt2-medium
    accelerate launch train.py --config configs/train_gpt_neo.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/logs/train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT / LLaMA model on custom text data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/train_gpt2.yaml)"
    )
    # Allow CLI overrides of any config field
    parser.add_argument("--model_name", type=str, default=None, help="Override model name")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override LR")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters")
    parser.add_argument("--load_in_4bit", action="store_true", help="4-bit quantization")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "command", nargs="?", choices=["prepare", "train"],
        default="train", help="Command to run"
    )
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load YAML config and apply any CLI overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model_name:
        config.setdefault("model", {})["name"] = args.model_name
    if args.dataset:
        config.setdefault("dataset", {})["name"] = args.dataset
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.num_train_epochs:
        config.setdefault("training", {})["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.gradient_checkpointing:
        config.setdefault("training", {})["gradient_checkpointing"] = True

    return config


def run_prepare(config: dict) -> None:
    """Download and preprocess dataset only."""
    from src.dataset import load_text_dataset, tokenize_dataset
    from src.model import load_tokenizer

    logger.info("=== DATA PREPARATION ===")
    tokenizer = load_tokenizer(config["model"]["name"])
    dataset = load_text_dataset(
        dataset_name=config["dataset"]["name"],
        text_column=config["dataset"].get("text_column", "text"),
        train_split=config["dataset"].get("train_split", "train"),
        eval_split=config["dataset"].get("eval_split", "validation"),
    )
    lm_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        text_column=config["dataset"].get("text_column", "text"),
        max_length=config["dataset"].get("max_length", 512),
    )
    # Save processed dataset
    output_path = "data/processed/" + config["dataset"]["name"].replace("/", "_")
    lm_dataset.save_to_disk(output_path)
    logger.info(f"Processed dataset saved to: {output_path}")


def run_train(config: dict, args: argparse.Namespace) -> None:
    """Full fine-tuning pipeline."""
    from src.dataset import load_text_dataset, tokenize_dataset
    from src.model import load_model, load_tokenizer
    from src.trainer import fine_tune

    logger.info("=== FINE-TUNING PIPELINE ===")
    logger.info(f"Model:   {config['model']['name']}")
    logger.info(f"Dataset: {config['dataset']['name']}")

    # 1. Tokenizer
    tokenizer = load_tokenizer(config["model"]["name"])

    # 2. Dataset
    processed_path = "data/processed/" + config["dataset"]["name"].replace("/", "_")
    if Path(processed_path).exists():
        from datasets import load_from_disk
        logger.info(f"Loading cached dataset from {processed_path}")
        lm_dataset = load_from_disk(processed_path)
    else:
        dataset = load_text_dataset(
            dataset_name=config["dataset"]["name"],
            text_column=config["dataset"].get("text_column", "text"),
            train_split=config["dataset"].get("train_split", "train"),
            eval_split=config["dataset"].get("eval_split", "validation"),
        )
        lm_dataset = tokenize_dataset(
            dataset,
            tokenizer,
            text_column=config["dataset"].get("text_column", "text"),
            max_length=config["dataset"].get("max_length", 512),
        )

    # 3. Model
    model = load_model(
        config["model"]["name"],
        use_lora=args.use_lora,
        load_in_4bit=args.load_in_4bit,
    )

    # 4. Fine-tune
    sample_prompts = config.get("generation", {}).get("sample_prompts", None)
    trainer = fine_tune(model, tokenizer, lm_dataset, config, sample_prompts=sample_prompts)

    # 5. Final evaluation
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    logger.info(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

    import math
    ppl = math.exp(eval_metrics.get("eval_loss", 0))
    logger.info(f"Final perplexity: {ppl:.2f}")


def main():
    # Ensure output dirs exist
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    args = parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config, args)

    if args.command == "prepare":
        run_prepare(config)
    else:
        run_train(config, args)


if __name__ == "__main__":
    main()
