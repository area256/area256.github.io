---
title: Pretraining a 0.5B model on one rented GPU
authors: [jp]
description: A 489M-parameter model trained from scratch on 2.1B tokens for about $8 so far, and what the numbers say.
---

We trained a 489M-parameter Llama-style model from scratch on a single rented RTX 5090. Two training phases are done, 2.1 billion tokens in all, and a third is running. This post covers what we built, how it scores, and one mistake worth avoiding.

## The setup

The model is a standard decoder-only transformer, written in plain PyTorch:

Parameters
: 489.3M (448.3M excluding embeddings)

Layers and width
: 24 layers, d_model 1280

Attention
: Grouped-query, 20 query heads sharing 4 key/value heads

Feed-forward
: SwiGLU, d_ff 3840

Normalization and positions
: Pre-norm RMSNorm, RoPE

Vocabulary
: 32,000-token byte-level BPE, trained on the same corpus

Context
: 2048 tokens

Batch
: 524,288 tokens per step
{: .facts}

The training data is 2.03B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), with 50M held out for validation. The GPU was a 32 GB RTX 5090 rented on Vast.ai at $0.479 an hour. It trained at 48,000 tokens a second, about 85% of the card's theoretical peak.

## Results so far

| | Phase 1 | Phase 2 |
|---|--:|--:|
| Training tokens | 1.05B | 2.10B |
| Validation loss | 3.323 | 3.139 |
| Zero-shot average, 9 tasks | 0.412 | 0.433 |
| Wall clock | 6.4 h | 6.4 h |

Doubling the tokens lowered validation loss by 0.18 and raised average accuracy by 2 points. Neither curve has flattened yet. At 2.1B tokens the model has seen about 4 tokens per parameter, a fifth of the Chinchilla-optimal ratio, so tokens are still the main thing holding it back.

Training was stable throughout: across 4,000 steps there were no loss spikes and no non-finite gradients, and gradient clipping stopped triggering after step 540.

## Against public models

We ran every model through the same [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks on the same machine. Before evaluating our checkpoints, we converted them to the Hugging Face Llama format and confirmed the converted model gives the same outputs as the original.

| Model | Training tokens | Zero-shot average |
|---|--:|--:|
| SmolLM2-360M | 4T | 0.587 |
| Qwen2.5-0.5B | 18T | 0.565 |
| Pythia-410M | 300B | 0.493 |
| **Ours** | **2.1B** | **0.433** |
| Pythia-410M, same token count | 2.1B | 0.348 |

At the same 2.1B tokens, we score 8.5 points above Pythia-410M. We're 6 points behind the fully trained Pythia-410M, which saw 143 times as much data.

SmolLM2-360M beats Qwen2.5-0.5B with fewer parameters and less than a quarter of the tokens. That gap comes from how its training data was chosen and mixed. We haven't worked on data selection at all yet, so it's the most promising next step once tokens stop being the bottleneck.

On MMLU (5-shot) the model scores 0.251, which is chance for a four-way multiple choice. It hasn't learned that kind of knowledge yet, so we leave MMLU out of the average.

## Restarting the learning-rate schedule has a cost

Each phase used a cosine schedule: warm the learning rate up, then decay it to a tenth of its peak. To continue after phase 1, we warmed the learning rate back up to 2e-4 and decayed it again.

<figure class="wide">
<div class="chart-scroll">
{% include charts/llm-0.5b-val-loss.svg %}
</div>
<figcaption>Validation loss from 0.5B tokens on. The shaded area is where phase 2 sits above phase 1's final loss. Phase 3 is still running; the dashed segment is its first 100 steps. Hover a point to see its exact value; on a phone, scroll the chart sideways.</figcaption>
</figure>

Raising the learning rate on a model that had just been annealed undid part of that annealing. Validation loss rose from 3.323 to 3.383 and took about 520 steps to get back below 3.323, over a quarter of phase 2. The shaded area in the chart is that cost. Phase 3 shows the same jump: 3.139 to 3.183 in its first 100 steps.

Phase 2 still finished well ahead, so the restart was worth doing. But a warmup-stable-decay schedule avoids the cost entirely: hold the learning rate constant and decay it only at the very end. It's the better choice whenever the total training budget isn't fixed in advance, which for us it usually isn't.

## What moved and what didn't

| Task | 262M tokens | 1.05B | 2.10B |
|---|--:|--:|--:|
| LAMBADA | 0.021 | 0.151 | **0.212** |
| SciQ | 0.397 | 0.610 | 0.628 |
| ARC-Easy | 0.332 | 0.415 | 0.447 |
| PIQA | 0.545 | 0.594 | 0.614 |
| HellaSwag | 0.260 | 0.282 | 0.298 |
| ARC-Challenge | 0.232 | 0.233 | 0.249 |

LAMBADA, which asks the model to predict the last word of a passage, rose tenfold, making it the most useful single sign of progress. HellaSwag and ARC-Challenge barely moved and sit near chance, and Pythia's checkpoints at the same token counts do the same. At this scale those two tell you little.

We also logged per-layer gradient norms, activation sizes, and prediction entropy throughout. None of them needed action this time, but each would have caught a problem long before it showed up in the loss curve.

## Cost

| Stage | Hours | Cost |
|---|--:|--:|
| Tokenizer and corpus preparation | 0.65 | $0.32 |
| Phase 1 | 6.4 | $3.08 |
| Phase 2 | 6.4 | $3.08 |
| Benchmarks, including public baselines | 2.6 | $1.24 |
| **Total so far** | **16.1** | **$7.72** |

Phase 3 should add about 12 hours and $5.75.

## What's next

Phase 3 runs to 4.2B tokens, a second pass over the corpus. It also keeps a moving average of the weights, so we can measure whether averaging helps at this scale. After that, the plan is to change the data mix (adding code and reference text, aimed at the tasks that are currently flat) and to switch to a warmup-stable-decay schedule for any run whose length isn't fixed in advance.
