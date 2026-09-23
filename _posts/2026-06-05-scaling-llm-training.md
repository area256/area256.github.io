---
date: 2026-06-05 12:00:00 +0000
title: Scaling LLM training
authors: [jp]
description: What it takes to train a language model on more than one GPU, from memory accounting to 5D parallelism, following Hugging Face's Ultra-Scale Playbook and checked against our own single-GPU run.
---

Training a language model on one GPU is mostly a question of whether it fits. Training on hundreds of GPUs adds a second question: how to keep them all busy while they wait on each other. Hugging Face's [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Tazi et al., 2025) is the most thorough public guide to that second question. It's built on more than 4,100 distributed training experiments on up to 512 GPUs, and it walks through every technique used to spread training across a cluster.

This post follows the playbook's structure and keeps its numbers, and uses our own [0.5B model]({% post_url 2026-09-22-pretraining-a-0.5b-model-on-one-gpu %}), trained on a single RTX 5090, as the worked example. That run never needed more than one GPU, which makes it a useful baseline: it shows exactly where memory goes before any of the techniques below are needed.

The playbook frames everything around three constraints that trade against each other. **Memory** is a hard limit: if a step doesn't fit, it doesn't run. **Compute efficiency** is how much of the GPU's arithmetic you actually use. **Communication** is the time GPUs spend exchanging data instead of computing. Almost every technique buys one of these with another.

## Start with the memory budget

Four things take GPU memory during training: the model's weights, their gradients, the optimizer's state, and the activations saved during the forward pass for use in the backward pass. On top of those, CUDA itself takes 1 to 2 GB, and some memory is always lost to buffers and fragmentation.

For the first three, the playbook counts bytes per parameter. In the common mixed-precision setup, the model computes in bf16 but keeps an fp32 "master" copy of the weights for the optimizer to update:

| Item | Bytes per parameter |
|---|--:|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam first moment, fp32 | 4 |
| Adam second moment, fp32 | 4 |
| **Total** | **16** |

Some frameworks also accumulate gradients in fp32, because bf16 loses small values; that adds 4 bytes, for 20. Mixed precision doesn't save memory on these items at all. What it saves is compute time and activation memory.

At 16 to 20 bytes per parameter, the numbers grow quickly:

| Model | Weights, gradients and optimizer | With fp32 gradient accumulation |
|---|--:|--:|
| 1B | 16 GB | 20 GB |
| 7B | 112 GB | 140 GB |
| 70B | 1,120 GB | 1,400 GB |
| 405B | 6,480 GB | 8,100 GB |

From 7B parameters up, these alone don't fit on an 80 GB H100, before a single activation is stored.

**Our run.** Our model keeps its weights in fp32 and lets PyTorch's autocast run the matrix multiplications in bf16, so there's no separate bf16 copy: 4 bytes of weights, 4 of gradients and 8 of Adam state, which is also 16 bytes per parameter. For 489.3M parameters that's 7.8 GB. The full checkpoint we save for resuming is 5.9 GB, which is 12 bytes per parameter: weights plus Adam's two moments, since gradients aren't saved. Checking the checkpoint size against this arithmetic is a quick way to confirm you're saving what you think you're saving.

**Profile it.** The playbook uses PyTorch's profiler with memory tracking to watch a training step. Activations climb through the forward pass. During the backward pass, gradients accumulate while activations are freed. Then the optimizer step runs. One detail catches people out: the optimizer's state is only allocated during the first optimizer step, so a run can finish step 1 and then run out of memory on step 2.

## Activations grow with sequence length

Activation memory depends on the batch, not just the model. The playbook's estimate for a standard transformer in mixed precision is

*L · s · b · h · (34 + 5 · a · s / h)* bytes,

where *L* is the number of layers, *s* the sequence length, *b* the micro-batch size, *h* the hidden size and *a* the number of attention heads. It grows linearly with batch size. The second term, which is the attention matrix stored for the backward pass, grows with the square of the sequence length. For short sequences activations are minor; from about 2,000 to 4,000 tokens they become the largest item in memory.

<figure class="wide">
<ul class="legend">
<li><span class="legend-key"></span>With FlashAttention</li>
<li><span class="legend-key k-other-line"></span>With the attention matrix stored</li>
</ul>
<div class="chart-scroll">
{% include charts/scaling-activation-memory.svg %}
</div>
<figcaption>Estimated activation memory for our 24-layer model with 4 sequences per micro-batch, from the playbook's formula. The dotted line is the 24 GB left on a 32 GB GPU after 7.8 GB of weights, gradients and optimizer state. Hover a point to see its value.</figcaption>
</figure>

The chart shows why our run fit at all. Our micro-batch is 4 sequences of 2,048 tokens. Storing the attention matrix would take about 49 GB, far more than the GPU has. The attention kernel we use, PyTorch's `scaled_dot_product_attention`, never writes that matrix out, in the style of FlashAttention, which leaves about 8.6 GB. Add 7.8 GB of weights, gradients and optimizer state and the estimate is 16.4 GB, against the 19.6 GB peak we measured. The remaining 3.2 GB is most likely the output logits, 4 × 2,048 × 32,000 values, plus working memory. The formula also assumes a standard feed-forward block and full multi-head attention, which ours doesn't quite match, so treat it as an estimate.

**Activation recomputation** trades compute for memory: throw some activations away in the forward pass and recompute them in the backward pass. Recomputing everything, keeping only each layer's input, amounts to an extra forward pass and typically costs 30 to 40% more compute. Selective recomputation drops only the attention activations, which are large but cheap to recompute. For GPT-3 175B, the playbook cites a 70% cut in activation memory for 2.7% more compute. FlashAttention already recomputes attention scores in the backward pass, so anyone using it is doing selective recomputation already.

Recomputation changes how efficiency should be counted. Hardware FLOPs utilization counts the recomputed work; model FLOPs utilization (MFU) counts only the work the model needs. The playbook prefers MFU, because what matters is how long training takes.

## Batch size and gradient accumulation

The playbook measures batch size in tokens. Large pretraining runs typically use between 4 million and 60 million tokens per batch: Llama 1 used about 4 million, and DeepSeek about 60 million. Sequence lengths of 2,000 to 8,000 tokens are typical, with longer contexts added near the end of training.

When the batch doesn't fit in memory, gradient accumulation runs several micro-batches in sequence and adds up their gradients before the optimizer step. Memory stays constant, but the micro-batches run one after another, so it's slower. Across GPUs, the global batch size is

*global batch = micro-batch × accumulation steps × data-parallel GPUs*.

The playbook's advice is to use more GPUs before more accumulation, since GPUs run in parallel and accumulation doesn't.

Our run used a micro-batch of 4 sequences, 64 accumulation steps and one GPU: 4 × 64 × 2,048 = 524,288 tokens per step. On 8 GPUs, the same batch would need only 8 accumulation steps per GPU.

## Data parallelism

Data parallelism is the simplest way to use more GPUs: every GPU holds a full copy of the model, processes a different slice of the batch, and the GPUs average their gradients with an all-reduce before the optimizer step. The playbook describes three optimizations that make it efficient:

- **Overlap communication with the backward pass.** Start averaging a layer's gradients as soon as they're computed, instead of waiting for the whole backward pass to finish.
- **Bucket the gradients.** Send them in a few large messages rather than many small ones.
- **Skip the averaging during accumulation.** Only average after the last micro-batch. Our training loop does this with PyTorch's `require_backward_grad_sync` flag.

Data parallelism alone doesn't reduce memory per GPU, since every GPU still holds everything. And it stops scaling well at around 512 GPUs, where the time for the all-reduce to go around the ring of GPUs can no longer be hidden behind computation.

**ZeRO** removes the duplication. Each stage shards one more item across the *N* data-parallel GPUs, with *Ψ* the parameter count and *k* = 12 bytes of optimizer state per parameter (fp32 master weights plus two Adam moments):

| Stage | Shards | Memory per GPU |
|---|---|---|
| Plain data parallelism | nothing | 2Ψ + 2Ψ + kΨ |
| ZeRO-1 | optimizer state | 2Ψ + 2Ψ + kΨ/N |
| ZeRO-2 | + gradients | 2Ψ + (2Ψ + kΨ)/N |
| ZeRO-3 (FSDP) | + weights | (2Ψ + 2Ψ + kΨ)/N |

For a 7B model on 8 GPUs, that's 112 GB per GPU with plain data parallelism, about 39 GB with ZeRO-1, 26 GB with ZeRO-2 and 14 GB with ZeRO-3, before activations. ZeRO-2 communicates no more than plain data parallelism, since an all-reduce is itself a reduce-scatter followed by an all-gather, and the playbook calls it usually the better choice over ZeRO-1. ZeRO-3 has to gather each layer's weights before using them, in both the forward and backward passes, which raises communication by about half. Fetching the next layer's weights while the current one computes hides most of that.

ZeRO can't shard activations, since each GPU's are different, and every layer still has to fit on one GPU while it runs. Those limits are what the next techniques address.

## Tensor and sequence parallelism

Tensor parallelism splits individual weight matrices across GPUs. A linear layer can be split by columns, where each GPU computes part of the output, or by rows, where each GPU computes a partial sum. The transformer's feed-forward block pairs a column split with a row split, so it needs only one all-reduce in the forward pass. Attention splits naturally by heads.

That communication sits in the middle of every layer's computation, so it can't be fully hidden, and it needs the fastest links available. The playbook's benchmarks show it clearly: going from 8 to 16 GPUs, which crosses from one node to two, cost tensor parallelism about 43% of its throughput, against 14% for pipeline parallelism. The rule is to keep tensor parallelism within a node, typically 8 GPUs. It also can't usefully exceed the number of attention heads. Our model has 20 query heads but only 4 key/value heads, so splitting it more than 4 ways would mean duplicating key/value heads.

Tensor parallelism leaves some activations whole: those of the normalization and dropout layers. Sequence parallelism splits those along the sequence dimension instead. It replaces each all-reduce with a reduce-scatter and an all-gather, which cost the same in total, and cuts the remaining activation memory by the tensor-parallel degree. With both, the playbook fits 16,000-token sequences for a 70B model across 16 GPUs.

## Context parallelism

At very long sequence lengths, 128,000 tokens and up, even full recomputation leaves too much activation memory. Context parallelism splits the sequence itself across GPUs, for every layer. Most of the model doesn't care: each token's feed-forward and normalization computations are independent. Only attention needs every token's keys and values.

**Ring Attention** handles that by passing key and value chunks around a ring of GPUs. Each GPU computes attention for its own queries against the chunk it holds, sends that chunk on, receives the next one, and repeats until it has seen them all, overlapping the transfer with the computation. A causal mask makes the naive split unbalanced, because early tokens attend to far less than late ones. Zig-zag attention fixes that by giving each GPU a mix of early and late tokens.

## Pipeline parallelism

Pipeline parallelism splits the model by layers: GPU 1 holds the first few layers, GPU 2 the next, and so on. It reduces weight memory per GPU, and it only sends activations between neighbouring stages, which makes it well suited to slower links between nodes. Its cost is the **bubble**: while the first micro-batch travels forward through the pipeline and back, most GPUs sit idle.

With *p* pipeline stages and *m* micro-batches per step, the idle fraction in the simplest schedule, all forward passes then all backward passes, is (*p* − 1) / *m*. More micro-batches shrink the bubble, but this schedule has to keep activations for all of them. The **1F1B** schedule alternates one forward and one backward pass once the pipeline is full, which gives the same bubble while keeping activations for only *p* micro-batches. **Interleaved** schedules give each GPU several non-adjacent chunks of layers; with *v* chunks per GPU the bubble shrinks to (*p* − 1) / (*v* · *m*), at the cost of *v* times more communication. Llama 3.1 used an interleaved 1F1B schedule.

The newest schedules go further. **Zero Bubble** splits the backward pass into the part that computes gradients for the layer's input, which the previous stage is waiting for, and the part that computes gradients for the weights, which can be scheduled later to fill gaps. **DualPipe**, used for DeepSeek-V3, feeds micro-batches into both ends of the pipeline at once and overlaps nearly all communication with computation.

## Expert parallelism

Mixture-of-experts models replace the single feed-forward block with many smaller "experts" and send each token to a few of them. Expert parallelism places different experts on different GPUs and routes tokens to them with an all-to-all exchange. No matrix is split, so it's lighter than tensor parallelism. DeepSeek-V3, for example, has 256 experts and limits each token to experts on at most 4 nodes, which bounds how far any token has to travel. Expert parallelism only splits the experts, so the rest of the model is usually combined with data parallelism.

## Choosing a configuration

The playbook calls the combination **5D parallelism**: data, tensor, sequence or context, pipeline and expert, with ZeRO on top of data parallelism.

| Method | Saves memory on | Splits along | Main cost |
|---|---|---|---|
| Data parallelism | nothing by itself | batch | limited by batch size |
| ZeRO-1, 2, 3 | optimizer state, then gradients, then weights | data-parallel copies | weight communication |
| Tensor and sequence | weights and activations | hidden dimension and sequence | needs very fast links |
| Context | activations | sequence | communication in attention |
| Pipeline | weights | layers | the bubble |
| Expert | expert weights | experts | routing |

Its procedure for choosing:

1. **Fit the model in memory.** Under about 10B parameters, one technique is often enough: tensor parallelism, or ZeRO-3 with full recomputation, across 8 GPUs. From 10B to 100B, combine tensor parallelism of 8 with pipeline parallelism or ZeRO-3. On more than about 512 GPUs, pure data parallelism or ZeRO-3 becomes inefficient, so add tensor or pipeline parallelism. Add context parallelism for long sequences and expert parallelism for mixture-of-experts models. With few GPUs, use full recomputation and more accumulation.
2. **Reach the target batch size** by adjusting data parallelism and accumulation.
3. **Maximize throughput.** Scale tensor parallelism up to the node size, then data parallelism with ZeRO-3, and switch to pipeline parallelism when data-parallel communication becomes the bottleneck. Change one dimension at a time.

Pipeline parallelism and ZeRO-3 both split weights across GPUs, but differently. Pipeline parallelism keeps whole layers and sends activations, and prefers many accumulation steps. ZeRO-3 keeps slices of every layer and sends weights, and prefers large micro-batches and long sequences. They're rarely used together, while ZeRO-1 or 2 combine easily with pipelines; DeepSeek-V3 used pipeline parallelism with ZeRO-1.

Across several thousand benchmarked configurations on 1 to 64 nodes of 8 H100s, efficiency fell as the number of nodes grew, especially for small models, because the fixed global batch of 1 million tokens left less work per GPU. The best MFU in the data behind the playbook's summary chart was about 44 to 47% for models from 1B to 9B parameters and about 34% for 80B. The playbook's most striking finding is that implementation quality mattered as much as the choice of technique: tensor parallelism beat pipeline parallelism until the pipeline code was optimized, after which pipeline parallelism won.

Our single-GPU run reached about 85% MFU, measured against the RTX 5090's rated bf16 throughput, at 48,000 tokens a second and about 3.7 billion floating-point operations per token. The two figures come from different hardware, so they aren't directly comparable. The gap does show what the playbook is about, though: most of the lost efficiency at scale is time spent communicating, and on one GPU there's nothing to communicate.

## Inside the GPU

Past the choice of parallelism, the playbook's last section is about making each GPU faster.

**Fuse operations.** A sequence of small element-wise operations, such as a normalization, reads and writes GPU memory at every step. A fused kernel does them all in one pass. `torch.compile` does much of this automatically, and our run used it. Custom kernels in Triton or CUDA go further at more effort.

**Use FlashAttention.** It computes attention in tiles held in the GPU's fast on-chip memory, never writing the full attention matrix out, which is the saving in the chart above. FlashAttention-2 roughly doubled its speed with better partitioning of work, and FlashAttention-3 adds support for newer hardware and fp8.

**Choose number formats deliberately.** The formats differ in how their bits are split between exponent, which sets the range, and mantissa, which sets the precision:

| Format | Sign / exponent / mantissa bits |
|---|---|
| fp32 | 1 / 8 / 23 |
| fp16 | 1 / 5 / 10 |
| bf16 | 1 / 8 / 7 |
| fp8, E4M3 | 1 / 4 / 3 |
| fp8, E5M2 | 1 / 5 / 2 |

bf16 keeps fp32's range with less precision, which is why it has largely replaced fp16 for training. fp8 doubles the throughput of bf16 on H100s but is harder to keep stable; DeepSeek-V3 was the first large public model trained with it, using separate scaling factors for small tiles of each matrix. The playbook still called fp8 training experimental in early 2025.

## Rules of thumb

- Weights, gradients and Adam state cost 16 bytes per parameter in standard mixed precision, 20 with fp32 gradient accumulation.
- Activations grow linearly with batch size and, without FlashAttention, with the square of sequence length.
- A run that survives step 1 and then runs out of memory has just allocated its optimizer state.
- Training costs about 6 floating-point operations per parameter per token: 2 in the forward pass and 4 in the backward pass.
- Use more GPUs before more gradient accumulation.
- Keep tensor parallelism inside a node, and no wider than the number of attention heads.
- Prefer ZeRO-2 to ZeRO-1; use ZeRO-3 when the weights themselves don't fit.
- Measure MFU, and change one parallelism dimension at a time.

## Sources

- Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro von Werra and Thomas Wolf, [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook), Hugging Face, 2025.
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra and Christopher Ré, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023.
