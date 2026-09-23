---
date: 2026-06-12 12:00:00 +0000
title: Scaling LLM serving
authors: [jp]
description: Why generating text is limited by memory rather than compute, and the techniques that serving systems use to get around it, from the KV cache to speculative decoding.
---

Training and serving a language model stress a GPU in opposite ways. Training processes millions of tokens at once and is limited by arithmetic, which is what our [post on scaling training]({% post_url 2026-06-05-scaling-llm-training %}) was about. Serving produces one token at a time for each request, and on most hardware it's limited by how fast the GPU can read its own memory. Almost every technique for serving models efficiently follows from that one fact.

This post works through those techniques in the order they build on each other. Where we can, it uses our own [0.5B model]({% post_url 2026-09-22-pretraining-a-0.5b-model-on-one-gpu %}) on an RTX 5090 as the worked example. Every source is listed at the end.

## Two phases: prefill and decode

A request to a language model runs in two phases. **Prefill** processes the whole prompt in one pass, all its tokens in parallel, like a training forward pass. **Decode** then generates the response one token at a time, and each new token needs a full pass through the model.

The two phases have different bottlenecks, and serving systems report them separately:

- **Time to first token** is mostly prefill. It's what a user feels as the pause before the answer starts.
- **Time per output token** is decode. It sets how fast the answer streams.
- **Throughput** is total tokens per second across all requests, which sets the cost.
- **Goodput**, a term from the DistServe paper, is the rate of requests that meet their latency targets. High throughput with most requests missing their targets isn't useful.

## Why decoding is limited by memory

To generate one token, the GPU has to read every weight in the model from memory. For one request at a time, each weight is used for only about two arithmetic operations: a multiply and an add. That's far too little work per byte to keep the GPU's arithmetic units busy.

kipply's [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) puts numbers on this. An A100 can do about 312 trillion bf16 operations a second but read only about 1.5 TB of memory a second, a ratio of roughly 208 operations per byte. Until each weight is used for about that many operations, the GPU spends its time waiting on memory, and processing one token takes about as long as processing a couple of hundred. Horace He's [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) gives the general framing: every operation is limited by compute, by memory bandwidth, or by overhead, and knowing which tells you what to optimize.

**Our model.** In bf16 our 489.3M weights take 0.98 GB. The RTX 5090 reads memory at about 1.8 TB/s, so one request at a time can never exceed about 1,800 tokens a second, however fast the arithmetic is. Its rated bf16 throughput of about 210 trillion operations a second is about 117 times its memory bandwidth in bytes, so it takes a batch of roughly 100 requests decoding together before arithmetic becomes the limit.

That's the central lesson of serving: **batch many requests together**, so each weight read from memory does work for all of them. The rest of this post is about how to fit a large batch in memory, keep it full, and waste as little of it as possible.

## The KV cache

Attention needs the keys and values of every earlier token. Recomputing them for the whole sequence at each step would make generating *n* tokens cost work proportional to *n*², so every serving system stores them instead. That's the **KV cache**.

Our own `generate` function, which we only use to print sample text during training, doesn't have one: it runs the whole sequence through the model for every new token. For a few 64-token samples every few hundred steps that's fine. For serving, the KV cache is the first thing to add.

The cache has a cost. Per token it takes 2 (keys and values) × layers × key/value heads × head size × bytes per value. kipply gives this for standard multi-head attention; with fewer key/value heads, the count of key/value heads replaces the count of attention heads. For our model in bf16 that's 2 × 24 × 4 × 64 × 2 = 24 KB per token, or about 50 MB for a full 2,048-token context.

<figure class="wide">
<ul class="legend">
<li><span class="legend-key k-other-line"></span>Multi-head, 20 KV heads</li>
<li><span class="legend-key"></span>Grouped-query, 4 KV heads (ours)</li>
<li><span class="legend-key k-2"></span>Multi-query, 1 KV head</li>
</ul>
<div class="chart-scroll">
{% include charts/scaling-kv-cache.svg %}
</div>
<figcaption>KV cache for one sequence of our model in bf16, computed from its shape. It grows in a straight line with context length. The dotted line is the size of the model's own weights. Hover a point to see its value.</figcaption>
</figure>

With standard multi-head attention, every one of our model's 20 attention heads would keep its own keys and values, five times as much. At 32K tokens, one sequence's cache would be about 4 GB, four times the size of the model itself. For a batch of requests, the KV cache, not the weights, is what fills the GPU.

**Share key/value heads.** Noam Shazeer's [multi-query attention](https://arxiv.org/abs/1911.02150) (MQA) observed that decoding is limited by reloading these cached keys and values, and let all attention heads share a single key/value head, shrinking the cache by the number of heads with little loss in quality. [Grouped-query attention](https://arxiv.org/abs/2305.13245) (GQA) is the middle ground: a few key/value heads, each shared by a group of query heads. Its authors found it reaches quality close to multi-head attention at speed close to multi-query attention, and that an existing multi-head model can be converted with about 5% of its original training compute. Our model uses GQA with 4 key/value heads, which is why its cache is 24 KB per token rather than 120 KB.

## Managing the cache: PagedAttention

Early serving systems reserved one contiguous block of memory per request, sized for the longest response it might produce. Most of that went unused. The [vLLM paper](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023) measured that in existing systems only 20 to 38% of the memory reserved for the KV cache held actual keys and values; the rest was lost to over-reservation and fragmentation.

**PagedAttention** borrows the operating system's answer to the same problem: virtual memory. It stores each request's cache in small fixed-size blocks that don't need to be contiguous, allocated as the sequence grows. Waste falls to nearly nothing, and requests can share blocks, for example when sampling several responses to the same prompt. The paper reports 2 to 4 times the throughput of the best earlier systems at the same latency. The launch [blog post](https://vllm.ai/blog/2023-06-20-vllm) put it at up to 24 times Hugging Face Transformers, with existing systems wasting 60 to 80% of their cache memory against under 4% for vLLM.

## Batching: static to continuous

With static batching, a batch starts together and finishes together: short responses wait for the longest one, and new requests wait for the whole batch. The [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) paper (Yu et al., 2022) replaced that with **iteration-level scheduling**: the batch is re-formed at every decoding step, finished requests leave immediately, and new ones join. This is now called **continuous batching**. Orca also batches only the operations that can be batched, such as the matrix multiplications, and runs attention per request, so sequences of different lengths can share a batch. On GPT-3 175B it reported 36.9 times the throughput of NVIDIA's FasterTransformer at the same latency.

Anyscale's [benchmark](https://www.anyscale.com/blog/continuous-batching-llm-inference) separates the two ideas. Serving OPT-13B on one A100 with widely varying response lengths, naive static batching produced about 81 tokens a second. Systems with continuous batching alone reached about 300 to 350, roughly 4 times more. vLLM, with continuous batching and paged memory, reached about 1,865, 23 times the naive baseline. Both ideas matter, and the memory management is worth more than the scheduling.

## Prefill and decode get in each other's way

Continuous batching mixes new requests' prefills with ongoing requests' decodes. A long prompt's prefill takes much longer than a decode step, so every request in the batch stalls while it runs, and output streaming stutters.

**Split long prefills into chunks.** [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (Agrawal et al., 2024) breaks each prompt into roughly equal chunks and adds one chunk per step alongside the ongoing decodes, so no step is ever much longer than the rest. Compared with vLLM, it reports 2.6 times the serving capacity for Mistral-7B on one A100, 3.7 times for Yi-34B on two, and 5.6 times for Falcon-180B.

**Or run them on different GPUs.** Prefill is limited by compute and decode by memory bandwidth, so the best hardware and parallelism differ. [DistServe](https://arxiv.org/abs/2401.09670) (Zhong et al., 2024) runs the two phases on separate GPUs, tunes each separately, and moves the KV cache between them. It reports serving 7.4 times more requests, or meeting latency targets 12.6 times tighter, than existing systems. [Splitwise](https://arxiv.org/abs/2311.18677) (Patel et al., 2024) makes the same split across machines, reporting 1.4 times the throughput at 20% lower cost, or 2.35 times at the same cost and power.

## Reuse shared prefixes

Many requests start the same way: a system prompt, few-shot examples, the earlier turns of a chat, the fixed preamble of an agent loop. Their KV cache for that shared start is identical. [SGLang](https://arxiv.org/abs/2312.07104) (Zheng et al., 2024) keeps cached prefixes in a tree, called RadixAttention, so any new request that shares a prefix picks up its cache instead of recomputing it. It reports up to 6.4 times the throughput of earlier systems on workloads such as agents, retrieval-augmented generation and multi-turn chat.

## Make attention itself cheaper

[FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) computes attention in tiles held in the GPU's small, fast on-chip memory, never writing the full attention matrix to main memory. That makes attention's extra memory grow linearly with sequence length instead of quadratically, and cuts the traffic to main memory, which is the resource decoding is short of. [FlashAttention-2](https://arxiv.org/abs/2307.08691) roughly doubled its speed by splitting the work better across the GPU. It's the same kernel that makes long-sequence training fit, as the training post shows.

## Make the weights smaller

If decoding is limited by reading the weights, fewer bytes per weight means faster decoding, as well as more room for the KV cache.

- **[LLM.int8()](https://arxiv.org/abs/2208.07339)** (Dettmers et al., 2022) runs models of up to 175B parameters in 8-bit with no loss in quality, halving memory. The difficulty was a few feature dimensions with very large values that appear in large models; it keeps those in 16-bit and runs about 99.9% of the multiplication in 8-bit.
- **[GPTQ](https://arxiv.org/abs/2210.17323)** (Frantar et al., 2022) quantizes weights to 3 or 4 bits after training, using approximate second-order information to correct for the error, with negligible accuracy loss. It quantizes a 175B model in about 4 GPU-hours and lets it run on a single GPU, about 3.25 times faster than fp16 on an A100.
- **[AWQ](https://arxiv.org/abs/2306.00978)** (Lin et al., 2023) observes that about 1% of weights matter most, and that they're best identified by the size of the activations they multiply rather than by their own size. Scaling those channels up before quantizing to 4 bits protects them, with no retraining.

For our model, 4-bit weights would take about 0.25 GB instead of 0.98 GB, which would raise the ceiling on single-request decoding speed about fourfold.

## Speculative decoding

Decoding one token barely uses the GPU's arithmetic, and checking several candidate tokens in one pass costs about the same as generating one. Speculative decoding exploits that. A small, fast draft model proposes the next few tokens, and the large model checks them all in a single pass, keeping the longest run it agrees with. A carefully designed acceptance rule makes the output follow exactly the same distribution as the large model alone, so quality is unchanged.

It was developed independently by two groups. [Leviathan et al.](https://arxiv.org/abs/2211.17192) at Google reported 2 to 3 times faster decoding for T5-XXL with identical outputs, and [Chen et al.](https://arxiv.org/abs/2302.01318) at DeepMind 2 to 2.5 times for the 70B-parameter Chinchilla, with no retraining in either case. It helps most at small batch sizes, where the GPU has arithmetic to spare.

## Spreading a model across chips

When a model is too big for one accelerator, it has to be split, and the best split depends on the batch size. [Pope et al.](https://arxiv.org/abs/2211.05102) (2022) built a model of the trade-offs for serving PaLM on TPUs. Layouts that keep weights in place and move activations win at small batches and low latency; layouts that move the weights instead win at large batches. With these choices they served the 540B-parameter PaLM at 29 ms per generated token using 8-bit weights, reached 76% utilization of the hardware during prefill of large batches, and used multi-query attention to support contexts up to 32 times longer.

## Rules of thumb

- Decoding is limited by memory bandwidth. One request at a time can't go faster than bandwidth divided by the size of the weights.
- Batching is the main lever: each weight read should serve as many requests as possible.
- At long contexts and large batches, the KV cache, not the weights, fills the GPU. Estimate it before choosing a batch size.
- Fewer key/value heads, through grouped-query or multi-query attention, shrinks the cache almost for free.
- Allocate the cache in pages and schedule at every step, not once per batch.
- Keep long prefills from stalling decodes, by chunking them or by running them on separate GPUs.
- Quantized weights and speculative decoding both help most when batches are small.
- Report time to first token and time per output token separately, and measure goodput, not just throughput.

## Sources

Background:

- kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/), 2022.
- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html), 2022.
- Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal and Jeff Dean, [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102), 2022.

Attention and the KV cache:

- Noam Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra and Christopher Ré, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023.
- Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón and Sumit Sanghai, [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.

Serving systems:

- Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim and Byung-Gon Chun, [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), 2022.
- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang and Ion Stoica, [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023, and the [vLLM launch post](https://vllm.ai/blog/2023-06-20-vllm), 2023.
- Cade Daniel, Chen Shen, Eric Liang and Richard Liaw, [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference), Anyscale, 2023.
- Lianmin Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), 2023.
- Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki and Ricardo Bianchini, [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677), 2023.
- Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin and Hao Zhang, [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670), 2024.
- Amey Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), 2024.

Faster decoding:

- Tim Dettmers, Mike Lewis, Younes Belkada and Luke Zettlemoyer, [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339), 2022.
- Elias Frantar, Saleh Ashkboos, Torsten Hoefler and Dan Alistarh, [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), 2022.
- Yaniv Leviathan, Matan Kalman and Yossi Matias, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2022.
- Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper, [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023.
- Ji Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), 2023.
