"""
evaluate.py
-----------
Evaluation metrics for text generation:
  - Perplexity
  - BLEU (sacrebleu)
  - ROUGE (rouge-score)
  - BERTScore
  - Distinct-N (diversity)
"""

import logging
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    stride: int = 512,
    max_length: int = 1024,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity over a list of texts using sliding window.

    Lower is better. A perfectly calibrated model would have perplexity
    equal to the branching factor of the data.

    Args:
        model:      Causal language model.
        tokenizer:  Corresponding tokenizer.
        texts:      List of reference texts.
        stride:     Sliding window stride (tokens).
        max_length: Maximum context length.
        batch_size: Evaluation batch size.
        device:     Compute device.

    Returns:
        Scalar perplexity value.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            encodings = tokenizer(text, return_tensors="pt")
            seq_len = encodings.input_ids.shape[1]

            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - begin_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100  # Mask non-target tokens

                outputs = model(input_ids, labels=target_ids)
                nll = outputs.loss * trg_len

                total_nll += nll.item()
                total_tokens += trg_len

                if end_loc == seq_len:
                    break

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(
    predictions: List[str],
    references: List[str],
    tokenize: str = "13a",
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU score using sacrebleu.

    Args:
        predictions: Generated texts.
        references:  Reference texts (same length).
        tokenize:    sacrebleu tokenization scheme.

    Returns:
        Dict with 'bleu', 'precisions', and 'bp' keys.
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("Install sacrebleu: pip install sacrebleu")

    refs = [[r] for r in references]  # sacrebleu expects list of reference lists
    result = sacrebleu.corpus_bleu(predictions, list(zip(*refs)), tokenize=tokenize)
    return {
        "bleu": result.score / 100.0,
        "bleu_1": result.precisions[0] / 100.0,
        "bleu_2": result.precisions[1] / 100.0,
        "bleu_3": result.precisions[2] / 100.0,
        "bleu_4": result.precisions[3] / 100.0,
        "brevity_penalty": result.bp,
    }


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: Generated texts.
        references:  Reference texts.

    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("Install rouge-score: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg: Dict[str, List[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in agg:
            agg[key].append(scores[key].fmeasure)

    return {k: float(np.mean(v)) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Compute BERTScore (semantic similarity).

    Args:
        predictions: Generated texts.
        references:  Reference texts.
        model_type:  BERT model for computing embeddings.
        batch_size:  Batch size for BERTScore computation.

    Returns:
        Dict with precision, recall, f1 (averaged over corpus).
    """
    try:
        from bert_score import score
    except ImportError:
        raise ImportError("Install bert-score: pip install bert-score")

    logger.info(f"Computing BERTScore with {model_type}...")
    P, R, F1 = score(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


# ---------------------------------------------------------------------------
# Distinct-N (diversity)
# ---------------------------------------------------------------------------

def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Compute Distinct-N: ratio of unique N-grams to total N-grams.

    Higher values indicate more diverse/less repetitive generation.

    Args:
        texts: List of generated strings.
        n:     N-gram order (1 or 2 typical).

    Returns:
        Distinct-N score in [0, 1].
    """
    all_ngrams: List[Tuple] = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    model,
    tokenizer,
    predictions: List[str],
    references: List[str],
    metrics: Optional[List[str]] = None,
    perplexity_texts: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run all requested evaluation metrics and return a combined results dict.

    Args:
        model:             The language model (for perplexity).
        tokenizer:         Tokenizer.
        predictions:       Generated text samples.
        references:        Reference texts for comparison.
        metrics:           List of metric names to compute.
                           Defaults to ['perplexity', 'bleu', 'rouge', 'distinct'].
        perplexity_texts:  Texts for perplexity computation (defaults to references).

    Returns:
        Dict mapping metric names to scores.
    """
    if metrics is None:
        metrics = ["perplexity", "bleu", "rouge", "distinct"]

    results: Dict[str, float] = {}

    if "perplexity" in metrics:
        logger.info("Computing perplexity...")
        ppl_texts = perplexity_texts or references
        results["perplexity"] = compute_perplexity(model, tokenizer, ppl_texts)
        logger.info(f"  Perplexity: {results['perplexity']:.2f}")

    if "bleu" in metrics:
        logger.info("Computing BLEU...")
        bleu = compute_bleu(predictions, references)
        results.update(bleu)
        logger.info(f"  BLEU: {bleu['bleu']:.4f}")

    if "rouge" in metrics:
        logger.info("Computing ROUGE...")
        rouge = compute_rouge(predictions, references)
        results.update(rouge)
        logger.info(f"  ROUGE-L: {rouge['rougeL']:.4f}")

    if "bertscore" in metrics:
        logger.info("Computing BERTScore...")
        bs = compute_bertscore(predictions, references)
        results.update(bs)
        logger.info(f"  BERTScore F1: {bs['bertscore_f1']:.4f}")

    if "distinct" in metrics:
        results["distinct_1"] = compute_distinct_n(predictions, n=1)
        results["distinct_2"] = compute_distinct_n(predictions, n=2)
        logger.info(f"  Distinct-1: {results['distinct_1']:.4f}")
        logger.info(f"  Distinct-2: {results['distinct_2']:.4f}")

    return results
