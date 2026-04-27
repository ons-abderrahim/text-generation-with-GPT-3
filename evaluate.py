#!/usr/bin/env python3
"""
evaluate.py
-----------
Evaluate a fine-tuned model on a HuggingFace dataset.

Usage:
    python evaluate.py \
        --model_path outputs/checkpoints/gpt2-ag-news \
        --dataset ag_news --split test \
        --metrics perplexity bleu rouge bertscore distinct
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned causal language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset ID or local path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to evaluate (for speed)")
    parser.add_argument("--metrics", nargs="+",
                        default=["perplexity", "bleu", "rouge", "distinct"],
                        choices=["perplexity", "bleu", "rouge", "bertscore", "distinct"],
                        help="Metrics to compute")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--prompt_prefix", type=str, default="",
                        help="Prefix to add to each sample for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    from src.model import load_model_and_tokenizer
    from src.evaluate import evaluate_all
    from src.generate import generate_text, GenerationConfig
    from datasets import load_dataset

    # Load model
    logger.info(f"Loading model: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} / {args.split}")
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Sample
    if len(ds) > args.num_samples:
        ds = ds.select(range(args.num_samples))
    texts = ds[args.text_column]
    logger.info(f"Evaluating on {len(texts)} samples")

    # Generate predictions (for BLEU/ROUGE/BERTScore)
    predictions = []
    if any(m in args.metrics for m in ["bleu", "rouge", "bertscore"]):
        logger.info("Generating predictions for comparison metrics...")
        gen_config = GenerationConfig(
            strategy="nucleus",
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.92,
        )
        # Use first ~50 tokens of each text as prompt, predict rest
        for text in texts[:100]:  # Limit for speed
            words = text.split()
            prompt = " ".join(words[:20]) if len(words) > 20 else text
            gens = generate_text(model, tokenizer, prompt, gen_config)
            predictions.append(gens[0] if gens else "")

        references = [" ".join(t.split()[20:]) for t in texts[:100]]
    else:
        predictions = []
        references = texts[:100]

    # Compute metrics
    results = evaluate_all(
        model=model,
        tokenizer=tokenizer,
        predictions=predictions or [""] * len(references),
        references=references,
        metrics=args.metrics,
        perplexity_texts=texts[:100],
    )

    # Display results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in sorted(results.items()):
        print(f"  {metric:25s}: {value:.4f}")
    print("=" * 50)

    # Save results
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({
                "model": args.model_path,
                "dataset": args.dataset,
                "split": args.split,
                "num_samples": len(texts),
                "metrics": results,
            }, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
