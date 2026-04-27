# Evaluation Metrics for Text Generation

## 1. Perplexity (PPL)

**What it measures**: How well the probability model predicts a test set.

```
PPL(W) = exp(-1/N · Σᵢ log P(wᵢ | w₁...wᵢ₋₁))
```

- Lower is better
- A perplexity of `k` means the model is as "confused" as if it had to choose uniformly from `k` options at each step
- Comparable only within the same tokenizer/vocabulary

**Typical values**: GPT-2 on WikiText-103 ≈ 29 PPL. After fine-tuning on domain data, expect significant reduction (15–20 PPL).

---

## 2. BLEU (Bilingual Evaluation Understudy)

**What it measures**: N-gram precision between generated and reference text.

```
BLEU = BP · exp(Σₙ wₙ · log pₙ)
```

where `pₙ` is the modified n-gram precision and `BP` is the brevity penalty.

- Higher is better (0–1 scale)
- BLEU-4 (4-gram) is the standard variant
- Limitation: doesn't capture semantic similarity

---

## 3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**What it measures**: Overlap of n-grams and longest common subsequences.

| Variant | Measures |
|---------|----------|
| ROUGE-1 | Unigram overlap |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest Common Subsequence |

- Higher is better (0–1 scale)
- ROUGE-L is most robust, capturing sentence-level structure

---

## 4. BERTScore

**What it measures**: Semantic similarity using contextual BERT embeddings.

```
BERTScore_F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

where Precision and Recall are computed by matching tokens via cosine similarity between BERT representations.

- Higher is better (typically 0.85–0.95 range)
- More robust than n-gram metrics for paraphrased or semantically equivalent text

---

## 5. Distinct-N (Diversity)

**What it measures**: Ratio of unique n-grams to total n-grams in generated text.

```
Distinct-N = |unique n-grams| / |total n-grams|
```

- Higher is better
- Low Distinct-N indicates repetitive, degenerate generation
- Useful as a complement to quality metrics

---

## Which Metric to Trust?

| Scenario | Recommended Metric |
|----------|-------------------|
| Training progress | Perplexity (fast, no references needed) |
| News / factual text | ROUGE-L + BERTScore |
| Creative text | Distinct-N + human eval |
| Translation-style tasks | BLEU-4 |
| Full evaluation suite | Perplexity + ROUGE-L + BERTScore + Distinct |
