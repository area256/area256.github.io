---
date: 2026-09-24 12:00:00 +0000
title: The Gemma 4 Developer Agent competition, day one
authors: [jp]
description: What Google DeepMind's Gemma 4 Developer Agent competition asks for, why every score starts near 10%, what we learned running Gemma 4 31B as a coding agent, and how we plan to fine-tune it.
---

Most coding-agent benchmarks let you bring the strongest model you can pay for. The [Gemma 4 Developer Agent competition](https://www.kaggle.com/competitions/gemma-4-developer-agent) on Kaggle fixes the model instead: every entry runs the same 4-bit Gemma 4 31B on four NVIDIA L4 GPUs, with no internet, and has to fix real Python bugs well enough to pass hidden tests. What you get to change is the agent around it: its prompts, its budgets, its structure, and fine-tuned LoRA adapters. The competition opened on September 23 and runs until December 2 with $65,000 in prizes, plus $35,000 for an optional paper track. After the first day the best public score is 0.12.

This post explains the competition from the ground up: what a task looks like, what a submission can and can't do, and why scores start so low. Then it goes through what we learned in the first 24 hours: replicating the evaluation harness locally, measuring Gemma on it, and comparing it with a model of the same size. We finish with the fine-tuning pipeline we built from that. Every source is listed at the end.

## The task

Each task is a real pull request from a Python repository, frozen at the commit just before the fix. The agent gets:

- The **pull request description** as its problem statement. It is often short and written for maintainers, not as a specification.
- A **sandbox** with the repository checked out in `/workspace`, its dependencies installed, and no network.
- **Nine tools**: run a shell command, read a file (150 lines at a time), edit a file by replacing a string, write a file, check the remaining budget, submit the patch, and three code-graph tools backed by precomputed call graphs and embeddings.

When the agent calls `submit_patch`, the harness takes `git diff` of the workspace, applies it to a fresh copy of the repository together with the maintainers' own tests from the pull request, and runs `pytest`. The task counts only if every test passes. The score is the fraction of tasks solved.

The 129 public development tasks come from fastapi (67), rich (48), requests (13) and httpx (1). The roughly 120 scored tasks come from **private repositories**, split evenly between the public and private leaderboards. The organizers kept only tasks that a frontier model could solve or nearly solve.

## What a submission can change

A submission is a zip of declarative [Google ADK](https://adk.dev/) configuration: YAML for the agents, Markdown prompts, sampling settings, a per-task budget file, optional ADK skills, and optional LoRA adapters. No Python from the submission ever runs on the host. That leaves these levers:

- **Prompts and structure.** One agent, or several: an agent can call another agent as a tool, and agents can be chained in sequences or loops.
- **Sampling and budgets.** Temperature, output length, and the per-task time, tool-call and turn limits.
- **LoRA adapters.** Up to eight, rank 128 or less, attached to specific agents, with the whole submission under 3 GiB. This is where post-training comes in.

The model is fixed: `gemma-4-31b-it-qat-w4a16-ct`, a 4-bit build of Google's quantization-aware-trained Gemma 4 31B, served by vLLM with a 32k-token context. All ~120 tasks must finish within 12 hours, and each team gets one submission a day.

## Why scores start near 10%

Almost every first-day entry scored 0.00 to 0.12. Several things stack up:

1. **The official sample configuration** gives each task one minute and ten tool calls, which is not enough to finish anything. Many early entries kept it.
2. **The tasks are pitched at frontier models**, and come from repositories the model has never seen.
3. **Grading is all or nothing.** The hidden tests check exact function names, error messages and output formats described in the pull request. A fix that is almost right scores zero.
4. **The time budget is tight.** Twelve hours for about 120 tasks leaves around five minutes each on four L4s.
5. **Scores move in big steps.** With about 60 tasks on the public leaderboard, one task is worth about 1.7 points, so 0.10 and 0.12 differ by a single task.

Our first submission, a single agent with a directive prompt and a five-minute, sixty-call budget, scored **0.10**, tied for second. Scoring took about eight of the twelve hours, so there is little room to give each task more time.

## Replicating the harness

The organizers published a detailed description of their harness but not the harness itself, so we rebuilt it: the nine tools with the same output limits, the same prompt construction, the same "please continue" nudges, patch extraction, and verification in a fresh copy of the repository. Two checks keep it honest:

- **Oracle run.** Applying each task's real fix must pass its tests. 110 of 129 public tasks pass locally; the other 19 need older library versions than our environments have.
- **Baseline run.** Applying no fix must fail. Three tasks passed without any change and were dropped, leaving 107 development tasks.

The model runs wherever we can get it: vLLM on a Kaggle TPU v5e-8 with the unquantized QAT weights (free, 20 hours a week, but often a two-hour queue), or an OpenAI-compatible API such as OpenRouter for quick checks.

## What Gemma 4 does as an agent

On the TPU, the single-agent baseline solved 9 of 29 and 11 of 30 development tasks in two runs. Variants we expected to help did not:

| Variant | Solved |
|---|---|
| Baseline, one agent | 9/29, then 11/30 |
| A read-only analyzer sub-agent for code search | 8/29 |
| Thinking disabled | 7/29 |
| Prompt fixes for the failures below | 8/30 |
| Prompt fixes plus temperature 1.0 | 9/30 |

The traces were more informative than the scores. Gemma either solves a task quickly (6 to 38 tool calls) or gets stuck and burns the whole budget. Two patterns dominate the stuck runs:

- **Failed edits.** In one run 48 of 77 `edit_file` calls failed because the text to replace contained backslashes the file didn't: Gemma copied `\"` where the source had `"`.
- **Loops.** One run issued the same failing shell command 39 times in a row.

A third problem is structural: several tasks per run crashed when the conversation grew past about 24,500 tokens. vLLM requires the prompt plus the space reserved for the reply to fit in 32,768 tokens, and our configuration reserved 8,192 for the reply (the official sample reserves 16,384, which caps the conversation even lower). ADK's context compaction only starts at 32,768, so it never gets a chance to run first.

To see whether this is Gemma or our harness, we ran [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B), an open model of similar size, through the identical harness and prompt. On the tasks both ran cleanly, Qwen solved 5 of 13 and Gemma 2 of 13, and Qwen made almost no failed edits. The tools work; Gemma's behavior with them is the bottleneck, and prompt changes alone didn't fix it. That points to changing the weights.

## The fine-tuning pipeline

The goal is supervised fine-tuning on successful agent runs, rendered exactly as the host renders them:

- **Data.** Gemma's own runs that passed, and public agent trajectories from an open-weight teacher: [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), generated by Qwen3-Coder-480B under Apache 2.0. We converted them to our nine tools and kept resolved runs that don't install packages, don't leave scratch files in the patch, don't depend on file views our read limit would cut, and fit in 28,000 tokens. We left out public trajectory sets generated by proprietary models, whose providers' terms restrict training competing models on their outputs. Whether the competition allows distillation from other models at all is still an open question on the forum; the organizers have said they are discussing it.
- **Rendering.** Each conversation is rebuilt exactly the way the host's ADK harness would send it, and passed through Gemma 4's own chat template the way vLLM applies it. Only the tokens the model generates itself receive loss.
- **Training.** QLoRA on the unquantized QAT weights, rank 64, adapting attention and MLP projections in the language model only. Gemma 4 31B's full-attention layers share one projection for keys and values; we leave that projection out on those layers, because a server may apply it differently from training.
- **Checking.** After training, the adapter is served on top of the exact 4-bit competition model in vLLM, as the host does, and must change the model's output. Some Gemma 4 adapters have been reported to load in vLLM without taking effect.

The first training set has 843 conversations and 18 million tokens, about six million of them trained. We expect one epoch to take two to three and a half hours on a single rented H100.

## What we're doing next

1. Train the first adapter and compare it with the baseline on 45 development tasks, served the way the host serves it.
2. Test the context fix, a smaller output reservation so conversations can grow to about 28,600 tokens, against the baseline on the same tasks.
3. If the adapter helps, collect more of Gemma's own successful runs on a larger pool of public tasks with runnable environments, retrain, and repeat. Later, train on pairs of passing and failing attempts at the same task.

We'll write up what holds up.

## Sources

**Competition**

- Google DeepMind, [The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent) on Kaggle: overview, rules, evaluation, data description and harness guide; and the [paper track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper).
- Kaggle discussion: [Clarification on distillation from external LLMs](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807).
- Google, [Gemma 4 on Kaggle Models](https://www.kaggle.com/models/google/gemma-4) (`gemma-4-31b-it-qat-w4a16-ct` and `gemma-4-31b-it-qat-q4_0-unquantized`).

**Software**

- Google, [Agent Development Kit (ADK)](https://adk.dev/) and its [agent config](https://adk.dev/agents/config/) format.
- [vLLM](https://github.com/vllm-project/vllm), its [Gemma 4 recipe](https://recipes.vllm.ai/Google/gemma-4-31B-it), and [vllm-tpu](https://pypi.org/project/vllm-tpu/) for TPU serving.
- [LiteLLM](https://github.com/BerriAI/litellm) and [OpenRouter](https://openrouter.ai/).

**Data**

- Nebius, [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) and [SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories).
- Jimenez et al., [SWE-bench](https://www.swebench.com/SWE-bench/).
- Pan et al., [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym); Yang et al., [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith-py).
