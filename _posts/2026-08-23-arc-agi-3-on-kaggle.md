---
date: 2026-08-23 12:00:00 +0000
title: ARC-AGI-3 on Kaggle
authors: [jp]
description: What the ARC Prize 2026 ARC-AGI-3 competition asks for, why the scores are so low, how the best public agent works, and where we think there is room to improve.
---

Most benchmarks hand a model a question and grade the answer. [ARC-AGI-3](https://arcprize.org/arc-agi/3) hands an agent a game it has never seen, with no instructions, and grades how quickly it learns to win. The [Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) built on it runs until November 2 with $850,000 in prizes, and with ten weeks to go the best public scores are still in the low single digits out of 100.

This post explains the competition from the ground up: what the environment looks like, how scoring works and why it's so harsh, what the constraints of a Kaggle submission mean in practice, and how the strongest public agent, Tufa Labs' Duck harness, plays. We finish with the weaknesses we found reading its code and what we plan to try. Every source is listed at the end.

## The environment

An ARC-AGI-3 game is a small interactive program. At every step the agent receives a **frame**: a grid of up to 64×64 cells, each holding one of 16 colors, plus some metadata (the current state, which actions are allowed, how many levels are complete). The agent replies with an **action**, the game updates, and a new frame comes back.

There are at most seven actions, and their meaning is never explained:

- `RESET` starts or restarts the game.
- `ACTION1` to `ACTION5` and `ACTION7` are simple actions. In many games they behave like up, down, left, right and "interact", but the agent has to find that out.
- `ACTION6` is a click: it takes an `(x, y)` coordinate on the grid.

Each game has several **levels** of increasing difficulty, and a game is in one of three states: `NOT_FINISHED`, `WIN` or `GAME_OVER`. Some games have a visible timer or move budget that ends the level when it runs out. Nothing tells the agent what the goal is. It has to work out the rules and the objective from how the frames change, then solve the level, then carry what it learned into the next one.

The competition ships 25 public games for development, and they give a sense of the range:

- They have between 6 and 10 levels each (median 7), 183 levels in total.
- 13 are tagged as using both keyboard-style actions and clicks, 7 as click-only and 4 as keyboard-only.
- Many actions return an animation rather than a single frame. When we stepped through every public game with random actions, 14 of the 25 regularly returned multi-frame responses, with up to 42 frames for one action, and in most of those the intermediate frames showed things the final frame didn't: a sliding block, a falling piece, a flash.

The scored games are different. Evaluation uses 110 private games the agent has never seen: 55 decide the public leaderboard and the other 55 the private one that sets the final ranking. Nothing learned about a specific public game helps directly, which is the point.

The [`arc-agi`](https://github.com/arcprize/arc-agi) toolkit runs games locally and offline, and the [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) repository provides a template agent with two methods to implement: `choose_action`, which picks the next action from the latest frame, and `is_done`, which decides when to stop.

## How scoring works

Scoring rewards two things: completing levels, and doing it with about as few actions as a person would. The [scoring methodology](https://docs.arcprize.org/methodology) works in three steps.

1. **Per level.** Each completed level is compared with a human baseline, the number of actions first-time human players needed. The level scores `(human actions / agent actions)²`, so matching the human count earns 1. Beating it earns little extra: the methodology caps a level at 1.15, and Kaggle caps overall scores at 100%. An uncompleted level scores 0.
2. **Per game.** Level scores are averaged with weights equal to the 1-indexed level number, so level 1 has weight 1, level 2 weight 2, and so on. The denominator is the sum of the weights of *all* levels in the game, finished or not.
3. **Overall.** The total is the average across games, reported as a percentage.

Two features of this formula explain most of what follows.

**The square punishes exploration.** An agent that needs three times as many actions as a person keeps only a ninth of that level's score. An agent that needs ten times as many keeps one percent. The human baselines are small: across the public games, level 1 needs a median of 30 actions (as few as 7 in one game, at most 78), and a whole game needs a median of 638. An agent that probes each action a few times to see what it does has already spent a good part of the budget for the first level.

**The weights reward depth.** In a 7-level game the weights sum to 28, so level 1 is worth 1/28 of the game, under 4%, and the last level is worth a quarter. Solving the first three levels at exactly human efficiency scores (1 + 2 + 3) / 28, about 21% of that game. Solving only level 1 at three times the human action count scores (1/9) / 28, about 0.4%.

So an agent that reliably solves a level or two of most games, slowly, lands where the leaderboard is now: around one percent. Moving the score means reaching deeper levels, and doing it efficiently.

## What a Kaggle submission can use

The competition is notebook-only, and the constraints shape every design decision:

- **Hardware.** One NVIDIA RTX Pro 6000 with 96 GB of memory. The competition started on H100s and [moved to the RTX Pro 6000 in May](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/695158); the machines are reserved for this competition's notebooks.
- **Time.** At most 9 hours for all 110 games, [raised from 6 hours](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944) with the hardware change. That's under five minutes of wall-clock per game if they ran one after another, so agents run many games in parallel.
- **No internet.** Models, wheels and code have to be attached as Kaggle datasets or models, and any model has to run locally. Frontier APIs are out.
- **Open weights.** Prize winners must release their code and use open models, so the practical choice is an open-weight model small enough to serve on one GPU.
- **One submission a day.** Each submission reruns the notebook against the private games, and the score comes back hours later.

There are also two milestone prizes, on June 30 and September 30, for the leaderboard leaders on those dates who have published their notebooks.

The combination is unusual. Unlike most Kaggle competitions there's no training set to fit and no labels, and unlike most agent benchmarks the agent can't call a frontier model. It has to get as much reasoning as possible out of a mid-sized open model in a fixed amount of GPU time.

## What has worked so far

Four earlier systems set the direction.

**Stochastic Goose.** Dries Smit's agent [won the ARC-AGI-3 agent preview competition](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db), which ran on a handful of preview games before the official release.

**ARCgentica.** [Symbolica's agent](https://www.symbolica.ai/blog/arc-agi-3) scored 36% the day after the official public games were released, using frontier models. It lets the model call itself recursively, in the style of [Recursive Language Models](https://arxiv.org/abs/2512.24601), so sub-agents can each work on part of a game with a limited number of actions. Its main lesson is that these models are good at solving problems by writing code, as in [CodeAct](https://arxiv.org/abs/2402.01030).

**RGB Agent.** The [RGB Agent](https://blog.alexisfox.dev/arcagi3) wraps a general-purpose coding harness, OpenCode, around the model instead of building an ARC-specific one. It keeps the full game history in a log file the model can search and read, which lets it analyze before acting and queue only a few actions at a time. It was the first to show near-human action efficiency on the three preview games.

**The Duck.** Tufa Labs' Duck harness, the successor to Stochastic Goose, [won the first milestone](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133) with a leaderboard score of 1.21% using Qwen 3.6 27B in FP8. Tufa published the notebook, the [code](https://github.com/Tufalabs/duck-harness) and a [blog post](https://tufalabs.ai/research/duck-harness), and most public notebooks since then are forks of it. It's worth understanding in detail.

## How the Duck plays

The Duck gives the model one tool: a Python interpreter preloaded with the game's state. There's no fixed perception pipeline and no hand-written planner. The model writes code to inspect the board and to act, and the harness runs it.

**The game as Python variables.** Each tool call starts a fresh interpreter with these variables already defined:

- `current_frame` and `previous_frame`, which expose `.ascii` (the grid as letters, one per color), `.segmentation` (the board split into connected single-color objects), the step and the level.
- `history` and `transitions`, the list of past actions and the frames before and after them.
- `valid_actions`, the actions allowed right now.
- `action(...)`, which executes one action or a whole list of them in the real game.

Actions get readable names (`UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, `MOUSE`), and because `action` can be called in a loop, the model can write a search, such as a breadth-first search to a target, and execute its result in one call. Each call has a 30-second limit and output is capped at 4,096 characters.

**A world model in notes.** The prompt asks the model to keep a running note with labeled sections: "World model:", "Goal model:", "Action model:", "Plan:" and so on. The harness pulls those sections out of each reply and repeats them in the next prompt, so the model's understanding survives even when older messages are dropped.

**A bounded context.** The full system prompt is always included. Older turns are evicted to keep the input around 32K tokens, which lets a game run for hours at a steady speed.

**One image per turn.** At the start of each turn the model also sees the current frame as an image, upscaled 4× to 256×256 pixels. Tufa found this works best with Qwen's 16×16-pixel image patches. They tried showing several frames or whole animations and found the model couldn't make use of them.

**Prompting did a lot of the work.** Tufa's write-up is candid that the prompt carries a lot of hard-won guidance. It tells the model not to treat an energy bar as the objective and not to invent goals like moving a block to a fixed position. It keeps the model from hallucinating robots or treating the game as an Atari title, and from printing whole boards into its own context. Custom tools didn't help: in Tufa's words, hand-crafting tools "seems to hinder the creative abilities of the model." The largest gains came from better base models and from adding the image.

On the 25 public games, over 20 runs each, the Duck averaged **1.60% ± 0.45**. That spread is the other thing to know about this competition: the same notebook can score very differently from one run to the next, and Tufa reported their own best submission scoring as low as 0.77% on a rerun. With one submission a day, a single leaderboard score says little about whether a change helped.

## Where we see room

We read through the public Duck notebooks and harness code to find what we'd change first. Three things stood out, and none of them needs a better model.

**Knowledge is thrown away too often.** The harness clears the world, goal and action notes whenever a level is completed, and also whenever the game ends in `GAME_OVER`. In the public notebooks a game over restarts the current level rather than the whole game, so the agent is back on the same level it was just learning, having forgotten what it found. Clearing notes at a level change is more defensible, since layouts change, but what each action does usually carries over. Because later levels carry most of the weight, anything that speeds up level 3 onward is worth more than it looks.

**Time is split evenly regardless of progress.** The public notebooks give every game the same fixed wall-clock budget. In the configurations we looked at, 28 games run at once with a little over two hours each, which in four waves just fits 110 games into 9 hours. A game stuck on level 1 after an hour holds its slot as long as a game that's clearing a level every twenty minutes. Moving time from stalled games to progressing ones, or starting queued games sooner, is a scheduling change that doesn't touch the agent at all.

**The model never sees animations.** The harness keeps only the last frame of each action. In more than half the public games, the frames in between show how the action worked: which way something moved, whether a projectile hit anything, where a piece landed before snapping into place. Tufa found that passing whole animations as images didn't help a model of this size. A compact text summary of what changed during the animation, added to the action's result, might.

There are also two smaller points. The prompt never tells the model how many actions it has spent on the current level, even though efficiency is squared in the score. And every run is sampled without a fixed seed, which adds to the run-to-run noise that makes changes hard to evaluate.

## What we're doing next

Our plan is to test those changes one at a time against an unmodified baseline, on the public games, before spending any daily submissions on them:

1. Keep the world, goal and action notes through a game over, and carry the action and goal notes into each new level, marked as needing a quick recheck.
2. Stop a game after it has gone about an hour without completing a level, and let games that are still progressing use the time freed up.
3. Add a short summary of each action's animation to its result.

Because a single run is so noisy, we'll compare games one by one between paired runs rather than looking only at the averages, and repeat runs where the result is close. We'll write up what holds up.

## Sources

**Competition and environment**

- ARC Prize, [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) on Kaggle: overview, rules, evaluation and data description.
- ARC Prize, [ARC-AGI-3](https://arcprize.org/arc-agi/3) and [documentation](https://docs.arcprize.org/), including the [scoring methodology](https://docs.arcprize.org/methodology).
- ARC Prize, [arc-agi toolkit](https://github.com/arcprize/arc-agi) and [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) on GitHub.
- Kaggle discussion: [Upgraded accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/695158) and [Update on code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944).

**Agents**

- Tufa Labs (Bessis, Cottaar, Pressman, Smit, Tešnar, Viel), [Tufa Labs' winning solution for ARC-AGI-3 Milestone 1](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133), the [duck-harness repository](https://github.com/Tufalabs/duck-harness) and the [blog post](https://tufalabs.ai/research/duck-harness).
- Dries Smit, [1st place in the ARC-AGI-3 agent preview competition](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db).
- Symbolica, [ARC-AGI-3](https://www.symbolica.ai/blog/arc-agi-3).
- [RGB Agent](https://blog.alexisfox.dev/arcagi3) (blog.alexisfox.dev).

**Background**

- Wang et al., [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030) (CodeAct).
- Zhang, Kraska and Khattab, [Recursive Language Models](https://arxiv.org/abs/2512.24601).
