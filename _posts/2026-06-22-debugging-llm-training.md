---
date: 2026-06-22 18:00:00 +0000
title: Debugging LLM training
authors: [jp]
description: A training run with a bug rarely crashes. It just ends up a little worse. Here is how we check for that, and what we saw in our own run.
---

Most bugs in ordinary code announce themselves: an exception, a failing test, a wrong answer on the screen. Bugs in neural network training usually don't. A wrong mask, a shifted label, or a learning rate ten times too high still produces a loss that goes down. You end up with a model that is somewhat worse than it should be, and nothing tells you so.

Andrej Karpathy made this point in [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), and it's the idea behind the two other guides this post draws on: Pierce Freeman's [Debugging Tips for Neural Network Training](https://pierce.dev/notes/debugging-tips-for-neural-network-training) and the debugging checklist from MIT's [How to AI (Almost) Anything](https://mit-mi.github.io/how2ai-course/spring2025/schedule/Debugging%20Tips.pdf) course.

For language models the stakes are higher because each attempt is expensive. One phase of our [0.5B pretraining run]({% post_url 2026-09-22-pretraining-a-0.5b-model-on-one-gpu %}) took more than six hours. A bug found after the run costs the whole run, so almost everything below is about finding problems before the expensive part starts, or in its first few minutes.

## Check the pieces before the run

**Know what the loss should be at step 0.** A freshly initialized model should be close to guessing uniformly, so its loss should be about the natural log of the vocabulary size. With our 32,000-token vocabulary that's ln(32,000) ≈ 10.37. A much higher starting loss usually means the output layer's initialization is off; a much lower one can mean the model is seeing the answer. We only started logging at step 10, where the loss was already 8.82, so we can't show our own step-0 value. Log it.

**Test that the model can't see the future.** An autoregressive model must predict each token from earlier tokens only. If the attention mask leaks, training loss looks wonderful and the model is useless at generation. Karpathy suggests checking this with gradients: the gradient of an output should only reach inputs it is allowed to depend on. Our test does the same check more simply: change the last token and confirm no earlier prediction moves.

```python
def test_causal_masking():
    """Changing a future token must not affect earlier logits."""
    model = Transformer(tiny_cfg()).eval()
    x = torch.randint(0, 100, (1, 8))
    y = x.clone()
    y[0, -1] = (x[0, -1] + 1) % 100
    with torch.no_grad():
        a, _ = model(x, x)
        b, _ = model(y, y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
```

**Overfit a single batch.** All three guides agree on this one. A model that can't drive its loss to near zero on one repeated batch has a bug in the model, the loss, or the optimizer step. Freeman suggests building up from one example to two, then five, so you see the capacity you expect before you scale up.

**Read the data after it's been processed.** Karpathy's first step is to spend real time with the raw data, and to look at it again at the last point before it enters the model. For an LLM that means decoding a batch of token IDs back into text. Tokenizer bugs, broken document boundaries and garbled encodings are obvious when you read them and invisible in a loss curve.

**Write it twice and compare.** Freeman's tip is to write a slow, obvious version of any tricky tensor code next to the fast one and check they agree. We used the same idea when converting checkpoints to the Hugging Face Llama format for evaluation: the export script runs both models on the same random tokens and fails if their outputs differ. The largest difference was 0.0039, which is bf16 rounding noise. Without that check, a transposed weight would have quietly lowered every benchmark score.

**Make runs and evaluations repeatable.** Fix the random seed so two runs of the same config match. Evaluate on the same data every time: our validation loss resets its sampler before each evaluation, so a change in validation loss reflects the model, not a different set of windows.

## Log more than the loss

Loss is one number summarizing millions of parameters. By the time it looks wrong, the cause usually started much earlier. Freeman recommends logging per-layer gradient and weight statistics so you can see a problem at its source. Our training loop records these every 50 steps:

| Signal | What going wrong looks like |
|---|---|
| Non-finite gradient count | Any value above zero: an overflow or a bad batch, usually right before the loss turns to NaN |
| Gradient norm per layer | One layer's norm growing or collapsing away from the rest |
| Fraction of steps clipped | Clipping on most steps late in training, which means the learning rate is too high |
| Weight update size, ‖Δw‖ / ‖w‖ | Not tracking the learning-rate schedule, or far larger in one layer |
| Activation size after each block | Growing without bound with depth or over time |
| Prediction entropy | Falling toward zero, which shows up as the model repeating itself |
| Loss spikes | Loss above its moving average by more than four standard deviations |

In our run these stayed quiet. There were no non-finite gradients in 4,000 steps, and clipping triggered on 21 of the 412 logged steps: 14 during the 200-step warmup and none after step 540. The update size followed the schedule: its median fell from 0.18 around step 550 to 0.03 by step 2050, then more than doubled when the learning rate restarted for phase 2. Our post on the 0.5B run has a [heatmap of the per-layer gradient norms]({% post_url 2026-09-22-pretraining-a-0.5b-model-on-one-gpu %}#training-health).

None of these needed action this time. That's the point: they cost almost nothing to log, and if something had gone wrong they would have shown where.

## Read what the model writes

Freeman suggests saving the model's predictions at regular intervals, and Karpathy suggests watching how they change. For a language model the easiest version is to generate text from a few fixed prompts every few hundred steps, always with the same settings, and read it. Here's what ours produced with greedy decoding:

| Step | Prompt | Continuation |
|--:|---|---|
| 200 | The history of | the United States of the United States of the United States… |
| 1000 | The history of | the United States is a fascinating one. It is a fascinating story of the American Revolution, the American Revolution… |
| 4000 | The history of | the United States is a fascinating one. The first American president was George Washington… |
| 4000 | Water boils at | a rate of about 1/2 inch per year… |
| 4000 | def fibonacci(n): | a small, usually white, glandular tissue that is located in the back of the neck… |

The first three rows show the usual progression: pure repetition, then fluent sentences that loop, then sentences that stay on topic. The last two are more useful. The model still can't state a simple fact, which matches its chance-level MMLU score. And it has never produced code for the code prompt. That's not a bug in training; our corpus is educational web text with almost no code. The samples show that more directly than any benchmark score, and code is first on our list of data to add.

## Match the symptom to the cause

The MIT checklist maps what you see in the loss curves to the likely cause. Adapted for pretraining:

| What you see | Likely cause | First thing to try |
|---|---|---|
| Loss becomes NaN | Overflow in fp16, bad input, or uninitialized weights | Switch to bf16; check the batch and the weights for NaNs |
| Loss barely moves | Learning rate too low, or the model isn't getting the gradient | Raise the learning rate; check gradient norms per layer |
| Loss is noisy or climbing | Learning rate too high | Lower it; add gradient clipping |
| Training loss far below validation | Overfitting, which in pretraining usually means too many passes over the data | More data, or fewer epochs |
| Validation loss suspiciously low | The validation data leaked into training | Deduplicate across the split |

One entry needs a pretraining caveat. In the first pass over the corpus, training and validation loss should sit close together because the model hasn't seen either. A gap only means something once the data starts repeating. Our phase 3 is a second pass, so that's when we'll watch it.

Not every jump is a bug, either. Our validation loss rose when phase 2 started, because we deliberately raised the learning rate again. Knowing what each change should do to the curves is what lets you tell an expected bump from a real problem.

## Change one thing at a time

Karpathy's advice for the model itself is to start from something known to work and resist being creative early. Copy a standard architecture, use Adam at a sensible learning rate, and add complexity one piece at a time, checking that each piece helps. We did exactly that: a standard Llama-style decoder and AdamW with a peak learning rate of 3e-4. It's also why our results can be compared with Pythia's at all. If we had changed the architecture and the data and the schedule at once, we couldn't say which change mattered.

Karpathy also warns against trusting a learning-rate schedule's defaults. We learned a version of that ourselves: restarting a cosine schedule to extend a run cost more than 500 steps of progress.

## The short version

Before the run:

- Loss at step 0 is close to ln(vocabulary size).
- Changing a future token doesn't change earlier predictions.
- The model drives its loss to near zero on one batch.
- A decoded batch reads as the text you expected.
- Anything written twice, such as a checkpoint conversion, gives matching outputs.
- Seeds are fixed and evaluation uses the same data every time.

During the run:

- Log per-layer gradient norms, update sizes and non-finite counts, not just the loss.
- Generate from fixed prompts on a schedule and read the output.
- Before reacting to a change in the curves, ask whether you caused it.

## Sources

- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), 2019.
- Pierce Freeman, [Debugging Tips for Neural Network Training](https://pierce.dev/notes/debugging-tips-for-neural-network-training), 2022.
- MIT MAS.S60, How to AI (Almost) Anything, [debugging checklist slides](https://mit-mi.github.io/how2ai-course/spring2025/schedule/Debugging%20Tips.pdf), spring 2025.
