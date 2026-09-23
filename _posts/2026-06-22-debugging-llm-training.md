---
date: 2026-06-22 18:00:00 +0000
title: Debugging LLM training
authors: [jp]
description: A training run with a bug rarely crashes. It just ends up a little worse. A checklist for finding those bugs, drawn from published training logs, papers and our own run.
---

Most bugs in ordinary code announce themselves: an exception, a failing test, a wrong answer on the screen. Bugs in neural network training usually don't. A wrong mask, a shifted label, or a learning rate ten times too high still produces a loss that goes down. You end up with a model that is somewhat worse than it should be, and nothing tells you so.

Andrej Karpathy made this point in [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), and it runs through every guide on the subject. This post collects what those guides, the published logbooks of large training runs, and a handful of papers say about finding these bugs in language models specifically. Where we can, we show what each check looked like in our own [0.5B pretraining run]({% post_url 2026-09-22-pretraining-a-0.5b-model-on-one-gpu %}). Everything we cite is listed at the end.

Language models raise the stakes in two ways. Each attempt is expensive: one phase of our run took more than six hours, and the runs behind PaLM and OPT took weeks on hundreds to thousands of accelerators. And some failures only appear at scale or late in training, long after the cheap checks have passed. So almost everything below is about finding problems before the expensive part starts, or in its first few minutes, and then noticing the rest as early as possible.

## Check the data and the tokenizer

**Read the data after it's been processed.** Karpathy's first step is to spend real time with the raw data, and to look at it again at the last point before it enters the model. For an LLM that means decoding a batch of token IDs back into text. Tokenizer bugs, broken document boundaries and garbled encodings are obvious when you read them and invisible in a loss curve.

**Deduplicate.** Web text repeats itself far more than you'd guess. [Lee et al.](https://arxiv.org/abs/2107.06499) found a single 61-word sentence repeated more than 60,000 times in C4. Removing duplicates cut the rate at which models reproduced training text verbatim by a factor of ten, and reached the same accuracy in fewer steps. Duplicates also cause instability: the [OLMo 2](https://arxiv.org/abs/2501.00656) team traced many of their loss spikes to documents containing the same short phrase repeated dozens of times, and filtering them out nearly eliminated the spikes.

**Check for overlap with your benchmarks.** Lee et al. also found that over 4% of some standard validation sets appeared in the training data. The [GPT-3 paper](https://arxiv.org/abs/2005.14165) flagged a benchmark example as contaminated if it shared any 13-token sequence with the training set, then compared scores on the clean subset. It also admits that a bug in their filtering left some overlap in place, and that retraining to fix it was too expensive. Run the check before training, while fixing it is still cheap.

**Look for tokens the model never learns.** A tokenizer trained on different data from the model can contain tokens that almost never appear in training. Their embeddings stay close to their random starting point, and the model behaves erratically when it meets them. The best-known case is [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation), one of about 140 GPT-2 and GPT-3 tokens, many of them Reddit usernames, that made the models evade, insult or hallucinate. [Land and Bartolo](https://arxiv.org/abs/2405.05417) show how to find such tokens automatically from the tokenizer and the model's output weights, and found them across many open models. We trained our tokenizer on a million documents from the same corpus as the model, which keeps the mismatch small, but a quick scan for rarely used tokens is still worth doing.

**Know where documents start and end.** Most pretraining pipelines pack documents end to end and cut fixed-length windows out of the stream, so a window can span the end of one document and the start of the next. Ours marks each document with start and end tokens, and attention can still reach across that boundary. That's a common and reasonable choice, but it's a choice. Make it deliberately, and make sure evaluation formats text the same way (more on that below).

## Check the model and the loss

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

The related bug is an off-by-one in the targets. Each target should be the input shifted left by one token. If inputs and targets line up exactly, the model only has to copy its input, and the loss falls far below anything plausible almost immediately.

**Overfit a single batch.** Every guide we read agrees on this one. A model that can't drive its loss to near zero on one repeated batch has a bug in the model, the loss, or the optimizer step. Pierce Freeman's [debugging tips](https://pierce.dev/notes/debugging-tips-for-neural-network-training) suggest building up from one example to two, then five, so you see the capacity you expect before you scale up.

**Scale the initialization with depth.** Each transformer block adds its output to a running sum, so with standard initialization the sum grows with the number of layers. GPT-2 and Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) shrink the initial weights of each block's output projection by 1/√(2 × layers) to compensate; we do the same, giving 0.02/√48 for our 24 layers. The BLOOM team found the usual 0.02 standard deviation too large at 100B+ parameters and used a much smaller value, as described in Stas Bekman's [ML Engineering](https://github.com/stas00/ml-engineering) book. [Takase et al.](https://arxiv.org/abs/2312.16903) show the other side: if the embeddings start too small relative to the rest, normalization layers amplify their gradients and training spikes. Scaling the embeddings up, or normalizing them, fixed it.

**Be deliberate about weight decay.** Weight decay on normalization gains and biases pulls them toward zero for no benefit. Our optimizer decays only weight matrices. OLMo 2 goes further and also leaves the embeddings undecayed. Either way, print the parameter groups once and check each parameter landed where you meant.

**Write it twice and compare.** Freeman's tip is to write a slow, obvious version of any tricky tensor code next to the fast one and check they agree. We used the same idea when converting checkpoints to the Hugging Face Llama format for evaluation: the export script runs both models on the same random tokens and fails if their outputs differ. The largest difference was 0.0039, which is bf16 rounding noise. Without that check, a transposed weight would have quietly lowered every benchmark score.

## Check the training loop

**Normalize the loss correctly across gradient accumulation.** When a batch is too big for memory, you split it into micro-batches and add up their gradients. For this to match the full batch, each micro-batch's loss has to be weighted by its share of the batch's tokens. In October 2024, [Hugging Face](https://huggingface.co/blog/gradient_accumulation) and [Unsloth](https://unsloth.ai/blog/gradient) described a bug in the widely used Transformers trainer: it averaged each micro-batch's mean loss, which gives short, padded sequences too much weight. Training with accumulation produced a noticeably higher loss than the same batch without it. Our loop simply divides each micro-batch's loss by the number of micro-batches, 64, as nanoGPT does. That's correct here only because every window has exactly 2,048 tokens and nothing is padded. For fine-tuning on variable-length examples, the same line would be the bug.

**Give each GPU its own random stream.** With several GPUs, each one must draw different data. We seed each rank's data sampler with the base seed plus its rank. Hugging Face's [Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/) describes the reverse mistake in the SmolLM3 run: every tensor-parallel rank shared one seed, so their slices of the weights started out correlated. The run fell behind a comparison run and had to be restarted after a trillion tokens.

**Make runs and evaluations repeatable.** Fix the random seed so two runs of the same config match. Evaluate on the same data every time: our validation loss resets its sampler before each evaluation, so a change in validation loss reflects the model, not a different set of windows.

**Use bf16 rather than fp16.** fp16 can't represent large values, and a single overflow turns the loss into NaN. fp16 training needs a loss scaler, which our loop enables only when fp16 is selected. The BLOOM team credits much of their final run's stability to switching to bf16 after their earlier fp16 attempts diverged, and the [MIT course checklist](https://mit-mi.github.io/how2ai-course/spring2025/schedule/Debugging%20Tips.pdf) gives the same advice for anyone seeing NaNs.

**Log gradient norms before clipping.** Clipping caps the gradient norm, so a norm logged after clipping can never show you a problem above the cap. Google's [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) suggests logging the unclipped norm and setting the clip threshold around the 90th percentile of what you observe. If most steps are clipped, the clip is doing the job the learning rate should be doing.

## Log more than the loss

Loss is one number summarizing millions of parameters. By the time it looks wrong, the cause usually started much earlier. Freeman recommends logging per-layer gradient and weight statistics so you can see a problem at its source. Our training loop records these every 50 steps:

| Signal | What going wrong looks like |
|---|---|
| Non-finite gradient count | Any value above zero: an overflow or a bad batch, usually right before the loss turns to NaN |
| Gradient norm per layer, before clipping | One layer's norm growing or collapsing away from the rest |
| Fraction of steps clipped | Clipping on most steps late in training, which means the learning rate is too high |
| Weight update size, ‖Δw‖ / ‖w‖ | Not tracking the learning-rate schedule, or far larger in one layer |
| Activation size after each block | Growing without bound with depth or over time |
| Largest output logit | Climbing steadily, the early sign of output-logit divergence |
| Prediction entropy | Falling toward zero, which shows up as the model repeating itself |
| Loss spikes | Loss above its moving average by more than four standard deviations |

In our run these stayed quiet. There were no non-finite gradients in 4,000 steps, and clipping triggered on 21 of the 412 logged steps: 14 during the 200-step warmup and none after step 540. The update size followed the schedule: its median fell from 0.18 around step 550 to 0.03 by step 2050, then more than doubled when the learning rate restarted for phase 2. The largest output logit stayed between 19 and 23 for the whole run. Our post on the 0.5B run has a [heatmap of the per-layer gradient norms]({% post_url 2026-09-22-pretraining-a-0.5b-model-on-one-gpu %}#training-health).

None of these needed action this time. That's the point: they cost almost nothing to log, and if something had gone wrong they would have shown where.

Throughput is worth watching too. It isn't a model metric, but a drop usually means something else is wrong. Our run held 85% of the GPU's peak throughput throughout. In the Smol Training Playbook, drops in throughput lined up with spikes in disk read latency.

## Understand loss spikes

A loss spike is a sudden jump in training loss, sometimes recovering by itself and sometimes sending the run off the rails. They're the most documented failure in large-scale pretraining, and the published accounts agree on more than you might expect.

**They aren't just bad data.** Training the 540B-parameter [PaLM](https://arxiv.org/abs/2204.02311) model hit about 20 spikes despite gradient clipping. Each time, the team restarted from a checkpoint about 100 steps before the spike and skipped the next 200 to 500 batches. When they replayed the same batches from a different checkpoint, no spike appeared. The cause was the combination of that data with that state of the model.

**Gradient norms warn first.** In OLMo 2, gradient-norm spikes came before loss spikes and became more frequent as models grew. That's another reason to log the unclipped norm.

**Many of them are reproducible at small scale.** [Wortsman et al.](https://arxiv.org/abs/2309.14322) showed that small models trained at high learning rates show the same two failures as large ones. In the first, attention logits grow until attention collapses onto a single token; normalizing queries and keys (qk-layernorm) fixes it. In the second, output logits drift upward; a small extra loss term that penalizes their overall size, called z-loss, fixes it. PaLM and Wortsman et al. weight z-loss at 1e-4; OLMo 2 uses 1e-5. The same paper suggests longer warmup and a smaller AdamW epsilon, whose default can become as large as the gradients themselves at scale. The practical upshot: sweep the learning rate on a small model, and see where it breaks, before committing to the big one.

**They can come from the data loader.** The SmolLM3 run in the Smol Training Playbook had spikes because its loader read sequences in order within each document, so one very long, unusual file could fill a whole batch. Shuffling at the sequence level fixed it.

**The fixes that recur.** Meta's [OPT-175B logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) records weeks of restarts. Along the way, the team lowered gradient clipping from 2.5 to 1.0, raised weight decay to 0.1, lowered Adam's β2 to 0.95, and cut the learning rate when gradient and activation norms began to drift. Those are the same values our run started with: clipping at 1.0, weight decay 0.1, β2 0.95, and bf16. They're now common defaults for GPT-style pretraining. The Tuning Playbook adds two cheap checks: sweep warmup length over several orders of magnitude, and run a few hundred steps at twice your chosen learning rate to see how close to the edge you are.

Our spike detector flags any logged loss more than four standard deviations above its moving average. It didn't fire in 4,000 steps, which at our scale and learning rate is what the literature above would predict.

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

## Evaluate the way you trained

**Format evaluation text like training text.** Our training documents all start with a start-of-text token, so our evaluations add it too. Leave it out and every benchmark is measured on inputs the model never saw in training. Check the same for end-of-text tokens, chat templates after fine-tuning, and whitespace handling in the tokenizer.

**Verify every conversion.** Benchmark harnesses usually load models in a standard format, so you're often evaluating a converted copy rather than the model you trained. The export check described above is what makes our scores trustworthy.

**Pick benchmarks that move at your scale.** Small models sit at chance on many popular benchmarks. HellaSwag and ARC-Challenge barely moved in our run, while LAMBADA rose from 0.02 to 0.21, so that's the one to watch between checkpoints. A benchmark stuck at chance only adds noise to an average.

**Choose baselines carefully.** Comparing against a public model at the same number of training tokens is a useful control, but check where that checkpoint was in its own training. Pythia's 1-billion-token checkpoint is still inside its learning-rate warmup, which makes it a weak comparison; its 2-billion-token checkpoint is a fairer one.

## Long runs: resuming, repeating data, and hardware

**Resume the whole state, not just the weights.** A resumed run needs the optimizer state, the step count for the learning-rate schedule, and the data position. Restarting AdamW without its state resets its running averages, and the first steps after the resume take much larger updates than they should. Restarting the data sampler with its original seed replays the exact windows the run began with. We reseed the sampler from the step number on resume to avoid that.

**Restarting a learning-rate schedule has a cost.** Karpathy warns against trusting a schedule's defaults. We learned a version of that when we extended our run: warming a cosine schedule back up to continue training undid part of the previous annealing and cost more than 500 steps of progress. A warmup-stable-decay schedule, which holds the learning rate constant and decays it only at the end, avoids this.

**An epoch isn't always a pass over the data.** Our loader samples random windows rather than walking through the corpus in order. After 1.03 "epochs" of tokens, the arithmetic of random sampling says about 36% of the corpus has never been seen, while about 28% has been seen twice or more. That's fine, but it means a train/validation gap can open before you reach the epoch count you expect. How much repetition hurts is well studied: [Muennighoff et al.](https://arxiv.org/abs/2305.16264) found that up to about four epochs of repeated data are nearly as good as fresh data, with returns falling off quickly after that.

**Plan for hardware failure.** Most of OPT's early restarts came from hardware problems, not modeling ones. At our scale the equivalent is simpler but just as real. Our rented machine's disk disappears when the instance is destroyed, so we copied the phase 1 weights to another machine as soon as it finished. We ran training inside tmux so a dropped SSH connection wouldn't kill it. And we checked that there was enough free disk space before starting a phase, since running out mid-save can corrupt the checkpoint you most need.

## Match the symptom to the cause

The MIT checklist and Josh Tobin's [Troubleshooting Deep Neural Networks](http://josh-tobin.com/troubleshooting-deep-neural-networks.html) both map what you see in the curves to the likely cause. Adapted for pretraining, with what we added from the sources above:

| What you see | Likely cause | First thing to try |
|---|---|---|
| Loss becomes NaN | Overflow in fp16, bad input, or uninitialized weights | Switch to bf16; check the batch and the weights for NaNs |
| Loss rises from the start | A sign error in the loss or the update | Test the loss on inputs with a known answer |
| Loss barely moves | Learning rate too low, or the model isn't getting the gradient | Raise the learning rate; check gradient norms per layer |
| Loss is noisy or climbing | Learning rate too high | Lower it; add gradient clipping; lengthen warmup |
| Sudden loss spikes | High learning rate with a particular batch, repeated text, or growing logits | Restart from before the spike and skip ahead; filter repeats; add qk-layernorm or z-loss |
| Gradient norm drifting upward | The run is edging toward instability | Lower the learning rate before the loss reacts |
| Loss jumps after resuming | Optimizer state or data position not restored | Check what the checkpoint actually contains |
| Throughput drops | Data loading or disk, not the model | Check disk and loader latency |
| Training loss far below validation | Overfitting, which in pretraining usually means too many passes over the data | More data, or fewer epochs |
| Validation loss suspiciously low | The validation data leaked into training, or the targets aren't shifted | Deduplicate across the split; check the target offset |

One entry needs a pretraining caveat. In the first pass over the corpus, training and validation loss should sit close together because the model hasn't seen either. A gap only means something once the data starts repeating, which with random sampling happens sooner than the epoch count suggests.

Not every jump is a bug, either. Our validation loss rose when phase 2 started, because we deliberately raised the learning rate again. Knowing what each change should do to the curves is what lets you tell an expected bump from a real problem.

## Change one thing at a time

Karpathy's advice for the model itself is to start from something known to work and resist being creative early. Copy a standard architecture, use Adam at a sensible learning rate, and add complexity one piece at a time, checking that each piece helps. We did exactly that: a standard Llama-style decoder and AdamW with a peak learning rate of 3e-4. It's also why our results can be compared with Pythia's at all. If we had changed the architecture and the data and the schedule at once, we couldn't say which change mattered.

Small runs are how you afford that discipline. Wortsman et al. show that many large-scale failures can be reproduced in small models, and the Tuning Playbook is built around short, cheap experiments that each answer one question. Test a change small, then scale it.

## The short version

Data and tokenizer:

- A decoded batch reads as the text you expected.
- Exact and near duplicates are removed, including long runs of repeated text.
- No benchmark example shares a 13-token sequence with the training data.
- No tokens in the vocabulary are so rare the model never learns them.

Model and loss:

- Loss at step 0 is close to ln(vocabulary size).
- Changing a future token doesn't change earlier predictions, and targets are shifted by one.
- The model drives its loss to near zero on one batch.
- Output projections are scaled down with depth, and the parameter groups for weight decay are what you intended.
- Anything written twice, such as a checkpoint conversion, gives matching outputs.

Training loop:

- Gradient accumulation weights each micro-batch by its share of tokens.
- Each GPU has its own random stream, and runs are repeatable from a seed.
- Training uses bf16, and gradient norms are logged before clipping.

During the run:

- Log per-layer gradient norms, update sizes, output-logit size and non-finite counts, not just the loss.
- Generate from fixed prompts on a schedule and read the output.
- Know in advance how you'll respond to a loss spike.
- Before reacting to a change in the curves, ask whether you caused it.

Evaluation and long runs:

- Evaluation text is formatted like training text.
- Checkpoints save the optimizer, schedule and data position, and live somewhere that survives the machine.

## Sources

Guides:

- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), 2019.
- Josh Tobin, [Troubleshooting Deep Neural Networks](http://josh-tobin.com/troubleshooting-deep-neural-networks.html), Full Stack Deep Learning, 2019.
- Pierce Freeman, [Debugging Tips for Neural Network Training](https://pierce.dev/notes/debugging-tips-for-neural-network-training), 2022.
- Varun Godbole, George E. Dahl, Justin Gilmer, Christopher J. Shallue and Zachary Nado, [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), Google Research, 2023.
- Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering), 2023 onward.
- MIT MAS.S60, How to AI (Almost) Anything, [debugging checklist slides](https://mit-mi.github.io/how2ai-course/spring2025/schedule/Debugging%20Tips.pdf), spring 2025.
- Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi et al., [The Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/), Hugging Face, 2025.

Training logs and reports:

- Susan Zhang, Stephen Roller, Naman Goyal et al., [OPT-175B chronicles and logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles), Meta AI, 2022.
- BigScience, [BLOOM-176B training chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md), 2022.
- Aakanksha Chowdhery et al., [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311), 2022.
- Team OLMo, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656), Allen Institute for AI, 2024.
- Andrej Karpathy, [build-nanogpt](https://github.com/karpathy/build-nanogpt), 2024.

Papers:

- Tom Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020.
- Katherine Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499), 2021.
- Jessica Rumbelow and mwatkins, [SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation), 2023.
- Niklas Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264), 2023.
- Mitchell Wortsman et al., [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322), 2023.
- Sho Takase, Shun Kiyono, Sosuke Kobayashi and Jun Suzuki, [Spike No More: Stabilizing the Pre-training of Large Language Models](https://arxiv.org/abs/2312.16903), 2023.
- Sander Land and Max Bartolo, [Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models](https://arxiv.org/abs/2405.05417), 2024.

Bug reports:

- Hugging Face, [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation), 2024.
- Unsloth, [blog post on the gradient accumulation bug](https://unsloth.ai/blog/gradient), 2024.
