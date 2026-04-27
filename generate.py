#!/usr/bin/env python3
"""
generate.py
-----------
Generate text from a fine-tuned model.

Usage:
    # Interactive
    python generate.py --model_path outputs/checkpoints/gpt2-ag-news \
        --prompt "Breaking news:" --strategy nucleus --max_new_tokens 300

    # Batch from file
    python generate.py --model_path outputs/checkpoints/gpt2-ag-news \
        --prompts_file data/prompts.txt --output_file outputs/generated_texts/results.jsonl
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
        description="Generate text with a fine-tuned causal language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for interactive generation")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to a text file with one prompt per line")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results as JSONL to this path")

    # Generation config
    parser.add_argument("--strategy", type=str, default="nucleus",
                        choices=["greedy", "beam", "topk", "nucleus", "contrastive"],
                        help="Decoding strategy")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--num_return_sequences", type=int, default=1)

    # Model loading
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    from src.generate import TextGenerator, GenerationConfig

    generator = TextGenerator.from_pretrained(
        args.model_path,
        use_lora=args.use_lora,
        load_in_4bit=args.load_in_4bit,
    )

    # Build config
    config = GenerationConfig(
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences,
    )

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        logger.info("No prompt provided. Enter prompts interactively (Ctrl+C to exit):")
        while True:
            try:
                prompt = input("\n> ").strip()
                if not prompt:
                    continue
                texts = generator.generate(prompt, config=config)
                print("\n--- Generated ---")
                for i, t in enumerate(texts, 1):
                    print(f"[{i}] {prompt}{t}")
            except KeyboardInterrupt:
                logger.info("\nExiting.")
                sys.exit(0)

    # Batch generation
    from src.generate import generate_text
    results = []
    for prompt in prompts:
        logger.info(f"Generating for: '{prompt[:60]}...'")
        texts = generate_text(generator.model, generator.tokenizer, prompt, config)
        for text in texts:
            result = {"prompt": prompt, "generated": text, "full": prompt + text}
            results.append(result)
            print(f"\n[PROMPT] {prompt}")
            print(f"[OUTPUT] {text}\n")

    # Save output
    if args.output_file and results:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(results)} results to {args.output_file}")


if __name__ == "__main__":
    main()
