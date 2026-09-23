---
date: 2026-09-22 12:00:00 +0000
title: Pretraining a 0.5B model on one rented GPU
authors: [jp]
description: A 489M-parameter model trained from scratch on 4.2B tokens for $13.70, what it scores, and two things that did not work.
---

We trained a 489M-parameter Llama-style model from scratch on a single rented RTX 5090. The run is finished: three phases, 4.19 billion tokens, $13.70 of GPU time. This post covers what we built, how it scores, and two things we tried that did not work.

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

## Results

| | Phase 1 | Phase 2 | Phase 3 |
|---|--:|--:|--:|
| Training tokens | 1.05B | 2.10B | 4.19B |
| Validation loss | 3.323 | 3.139 | 2.998 |
| Zero-shot average, 9 tasks | 0.412 | 0.433 | 0.445 |
| MMLU, 5-shot | | 0.254 | 0.272 |
| Wall clock | 6.4 h | 6.4 h | 12.2 h |

Each doubling of the tokens paid, and each paid a little less than the one before. The first took validation loss down 0.18 and average accuracy up 2 points; the second, 0.14 and 1.2 points. Neither curve has bent toward a floor. At 4.2B tokens the model has seen about 8.5 tokens per parameter, still under half the Chinchilla-optimal ratio. Tokens were the limit from start to finish. We stopped because the budget ran out, not because the model stopped learning.

<figure class="wide">
<div class="chart-scroll">
{% include charts/llm-0.5b-loglog.svg %}
</div>
<figcaption>Validation loss against tokens with both axes logarithmic. A straight line here means the model is still in the regime where each doubling of data buys a fixed fraction of the remaining loss. Ours is still straight at the end, which is why we say the budget stopped the run rather than the model. Hover a point for its exact value.</figcaption>
</figure>

Training was stable throughout: across all 8,000 steps there were no loss spikes and no non-finite gradients, and gradient clipping stopped triggering after step 540.

<figure class="wide">
<div class="chart-scroll">
{% include charts/llm-0.5b-throughput.svg %}
</div>
<figcaption>Throughput and model-FLOPs utilisation over the run. It held 48,000 tokens a second at 85% of the card's theoretical peak from the first step to the last. The dips are benchmark runs sharing the GPU with training, which is the price of measuring as you go.</figcaption>
</figure>

## Against public models

We ran every model through the same [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks on the same machine. Before evaluating our checkpoints, we converted them to the Hugging Face Llama format and confirmed the converted model gives the same outputs as the original.

<figure class="wide">
<ul class="legend">
<li><span class="legend-key"></span>Our model, at each checkpoint</li>
<li><span class="legend-key k-2"></span>Pythia-410M</li>
<li><span class="legend-key k-other"></span>Other public models</li>
</ul>
<div class="chart-scroll">
{% include charts/llm-0.5b-accuracy.svg %}
</div>
<figcaption>Average zero-shot accuracy on nine tasks against training tokens, on a log scale. Pythia-410M was only measured at the three points shown; the dotted line just joins them. At 1.07B tokens, Pythia-1B and Pythia-410M score almost the same (0.298 and 0.299), so their dots overlap.</figcaption>
</figure>

<details markdown="1">
<summary>Show the numbers</summary>

| Model | Training tokens | Zero-shot average |
|---|--:|--:|
| SmolLM2-360M | 4T | 0.587 |
| Qwen2.5-0.5B | 18T | 0.565 |
| Pythia-410M | 300B | 0.493 |
| **Ours** | **4.19B** | **0.445** |
| Pythia-410M | 2.1B | 0.348 |
| Pythia-1B | 1.07B | 0.298 |

</details>

At a matched 2.1B tokens, we scored 8.5 points above Pythia-410M. Finishing at 4.19B puts us 9.7 points above that checkpoint and 4.8 points behind the fully trained Pythia-410M, which saw 72 times as much data.

SmolLM2-360M beats Qwen2.5-0.5B with fewer parameters and less than a quarter of the tokens. That gap comes from how its training data was chosen and mixed. We haven't worked on data selection at all yet, so it's the most promising next step once tokens stop being the bottleneck.

On MMLU (5-shot) the model finished at 0.272, up from 0.254 at half the tokens. Chance is 0.25 on a four-way multiple choice, so this is the first sign of that kind of knowledge appearing, and not much more than a sign: social sciences carries it at 0.320 while humanities is still at chance. We leave MMLU out of the average.

## Restarting the learning-rate schedule has a cost

Each phase used a cosine schedule: warm the learning rate up, then decay it to a tenth of its peak. To continue after phase 1, we warmed the learning rate back up to 2e-4 and decayed it again.

<figure class="wide">
<div class="chart-scroll">
{% include charts/llm-0.5b-val-loss.svg %}
</div>
<figcaption>Validation loss from 0.5B tokens on, with the learning rate underneath on the same axis. The shaded areas are where each phase sits above the previous phase's final loss. Hover a point to see its exact value; on a phone, scroll the chart sideways.</figcaption>
</figure>

Raising the learning rate on a model that had just been annealed undid part of that annealing. Validation loss rose from 3.323 to 3.383 and took 600 steps to get back below 3.323, about a third of phase 2. The shaded areas in the chart are that cost.

Phase 3 repeated it and showed what governs the size of the bill. It rose from 3.139 to 3.193, then needed 1,100 steps to recover, nearly twice phase 2's 600. The restart itself wasn't worse; phase 3's cosine simply runs over 4,000 steps instead of 2,000, so it holds the learning rate near its peak for twice as long. The cost tracks how long the schedule stays hot, not the act of restarting.

Both phases finished well ahead, so both restarts were worth doing. But together they spent 1,700 of the run's 8,000 steps, 21% of the budget, re-earning ground already taken. A warmup-stable-decay schedule avoids this entirely: hold the learning rate constant and decay it only at the very end. It's the better choice whenever the total training budget isn't fixed in advance, which for us it usually isn't.

## What moved and what didn't

<figure class="wide">
<ul class="legend">
<li><span class="legend-key"></span>Our model</li>
<li><span class="legend-key k-target"></span>Pythia-410M after 300B tokens</li>
<li><span class="legend-key k-chance"></span>Chance</li>
</ul>
{% include charts/llm-0.5b-tasks.html %}
<figcaption>Accuracy on each task against training tokens, all on the same 0 to 0.8 scale. The number beside each name is the final score at 4.19B tokens. LAMBADA asks for a free-text word, so it has no chance line.</figcaption>
</figure>

<details markdown="1">
<summary>Show the numbers</summary>

| Task | 262M tokens | 524M | 786M | 1.05B | 2.10B | 4.19B |
|---|--:|--:|--:|--:|--:|--:|
| LAMBADA | 0.021 | 0.107 | 0.141 | 0.151 | 0.212 | 0.230 |
| SciQ | 0.397 | 0.573 | 0.580 | 0.610 | 0.628 | 0.660 |
| ARC-Easy | 0.332 | 0.387 | 0.404 | 0.415 | 0.447 | 0.458 |
| PIQA | 0.545 | 0.573 | 0.589 | 0.594 | 0.614 | 0.628 |
| HellaSwag | 0.260 | 0.270 | 0.277 | 0.282 | 0.298 | 0.326 |
| ARC-Challenge | 0.232 | 0.224 | 0.231 | 0.233 | 0.249 | 0.265 |

</details>

LAMBADA, which asks the model to predict the last word of a passage, rose elevenfold from 0.021 to 0.230, making it the most useful single sign of progress. SciQ nearly doubled. HellaSwag sat nearly still for two phases and then finally began to move in phase 3, from 0.298 to 0.326.

ARC-Challenge ended at 0.265 and WinoGrande at 0.502, both at or near chance after the whole run, and Pythia's checkpoints at the same token counts behave the same way. At this scale those two cost evaluation time without informing a single decision. We'd drop them from the tracking suite and watch validation perplexity and LAMBADA instead.

## Averaging the weights didn't help

Phase 3 also kept an exponential moving average of the weights, on the usual reasoning that averaging away the noise the last steps leave in each parameter is worth a few tenths of a point. It wasn't. The averaged weights scored 0.443 against the final checkpoint's 0.445, and were slightly worse on validation perplexity too, 42.5 against 42.1. That's within noise, but it is certainly not a gain.

The reason makes sense in hindsight. Averaging helps when training ends while the learning rate is still high and the weights are still being jostled around. This run ends at 1e-5 after a full cosine decay, so the last thousand steps are already taking tiny, quiet steps. The decay had done the averaging already. The two techniques are substitutes, not complements.

So: if you can afford to decay the learning rate properly, do that and skip the averaging. Keep it for runs that end while the rate is still high, which is what happens with a constant-rate schedule, an early stop, or a budget that runs out mid-decay. Finding this out cost one gigabyte of disk and no measurable training time, which is a fair price for knowing.

## Training health

We also logged per-layer gradient norms, activation sizes, and prediction entropy throughout. None of them needed action this time, but each would have caught a problem long before it showed up in the loss curve.

<figure class="wide">
<div class="chart-scroll">
{% include charts/llm-0.5b-grad-heatmap.svg %}
</div>
<figcaption>Gradient norm for each of the 24 layers, every 100 steps, before clipping. Stronger color means a larger gradient. Hover a column to see its values.</figcaption>
</figure>

The first layer carries the largest gradients throughout, four to six times those of the early-middle layers, and the second half of the network runs higher than the first. All 24 layers shrink together, and each learning-rate restart shows up as a faint band at steps 2000 and 4000. No layer ever drifted away from the rest across the whole run. A layer whose gradients grew or collapsed on its own would be the first sign of an unstable run, and nothing here ever needed us to intervene.

<figure class="wide">
<div class="chart-scroll">
{% include charts/llm-0.5b-resid-heatmap.svg %}
</div>
<figcaption>Size of the residual stream leaving each layer, on the same sampling. This one is banded by depth rather than by time, so the pattern to look for is different.</figcaption>
</figure>

The residual stream tells a different story: it's banded by depth, not by time. Each layer adds its output to a running sum that only gets normalised on the way into the next block, so the stream grows steadily from layer 0 at the bottom to layer 23 at the top. That gradient is the signature of a healthy pre-norm network. What you don't want is a single row suddenly brightening, which is what a layer heading for numerical trouble looks like well before the loss notices.

<figure class="wide">
<div class="chart-scroll">
{% include charts/llm-0.5b-diagnostics.svg %}
</div>
<figcaption>Two more signals over the run. Top: mean prediction entropy, how uncertain the model is about its next token. Bottom: how large each optimizer update is relative to the weight it changes, for the embedding table and three attention projections, on a log scale.</figcaption>
</figure>

Prediction entropy drops steeply during warmup, from 7.44 nats to about 3.5 by the end of phase 1, and then stops falling. It spends phases 2 and 3 wandering around 3 nats, bouncing half a nat either way because we measure it on a single batch. That plateau is worth noticing: the model kept getting better at predicting the right token long after it stopped getting more *confident* on average. Loss and confidence are not the same thing, and only one of them was still improving.

The update sizes shrink as the learning rate decays and step back up at each restart, which is the loss curve's story told from the optimizer's side. None of this needed action. That's rather the point of logging it: it's cheap, and the run where one of these goes wrong is the run where you want it already there.

## Cost

| Stage | Hours | Cost |
|---|--:|--:|
| Tokenizer and corpus preparation | 0.65 | $0.32 |
| Phase 1 | 6.4 | $3.08 |
| Phase 2 | 6.4 | $3.08 |
| Phase 3 | 12.2 | $5.83 |
| Benchmarks, including public baselines | 2.9 | $1.39 |
| **Total** | **28.6** | **$13.70** |

## What we'd do next

The run is done, and it never hit a wall we could fix with engineering. Ranked by what would actually pay:

**Change the data mix, don't just add more of it.** The cheap token gains are largely spent, and the corpus is the binding constraint now. The benchmarks that stayed flat are flat because FineWeb-Edu contains none of what they test: no code, little reference text, nothing requiring multi-step reasoning. SmolLM2-360M beating Qwen2.5-0.5B on a quarter of the tokens is the evidence that mixture outweighs volume from here.

**If you do add tokens, use one warmup-stable-decay schedule.** Another doubling would still pay, since the curve is straight to the end. Running it as a single schedule recovers the fifth of the budget our two restarts cost.

**Skip the weight averaging if you can decay properly**, per the measurement above.

**Drop ARC-Challenge and WinoGrande from the tracking suite** at this scale. Both sat at chance for the whole run.

What we wouldn't bother with at this budget: architecture changes, a different optimizer, or a hyperparameter sweep. Nothing in the diagnostics ever suggested the model was the limiting factor. No instability, no dead layers, no gradient pathology, 85% of the card's peak throughput from the first step to the last. It was the data all along.
