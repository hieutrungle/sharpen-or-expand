# 006 — Thinking mode (amends 004) and a format-only halt (supersedes 005 item 6)

- Date: 2026-10-27
- Status: accepted (user, 2026-10-27: "enable_thinking=True" and "keep 16K; halt on format only",
  both the recommended options, chosen after the format probe below)
- Milestone: M0 (task M0.6)
- Frozen in: `soe/trainer/config/eval_suite.yaml` (`prompt.enable_thinking: true`) and
  `examples/data_preprocess/m0_hard_subset.yaml` (`min_boxed_rate_when_stopped: 0.8`)

## Context
Decision 004 fixed "the model's chat template" without naming its thinking mode. Qwen3.5's
generation prompt ends with an empty `<think>\n\n</think>\n\n` block by default and with an open
`<think>\n` when `enable_thinking=true`. Decision 005 item 6 halts the selection if fewer than 80%
of samples contain a `\boxed{}` answer, meant to catch format failures.

A format probe (`scripts/format_probe.py`, jobs 3221054 and 3221182 on one A40; samples under
`$SOE_DATA/probes/`) ran Qwen3.5-4B-Base on 200 RL-train problems (seeded draw; never evaluation
data, no Level A/B problem was sampled) × 4 samples, seed 1, suite sampling, in both modes:

| | empty think (default) | `enable_thinking=true` |
| --- | --- | --- |
| accuracy (truncated = incorrect) | 59.5% | 58.5% |
| boxed answer, all samples | 68.6% | 67.2% |
| truncated at 16,384 tokens | 34.3% | 36.0% |
| stopped samples: boxed / accurate | 100% / 90.5% | 100% / 91.4% |
| completions that write their own `</think>` | 63% | 64% |
| truncated samples in a repetition loop (heuristic) | 23% | 21% |

Paired difference (on − off): −1.0 ± 1.6 points (36 problems better, 45 worse, 119 equal).
Truncation is long, unfinished reasoning, not broken format: every stopped sample is boxed.

## Decision
1. **Thinking mode: `enable_thinking: true`** for every stage. The prompt ends with
   `<|im_start|>assistant\n<think>\n`. Every renderer passes it explicitly: evaluation (the
   suite), verl rollouts (chat-template kwargs, M1) and the SFT data format (M2:
   `<think>reasoning</think> answer`). The base model already writes in this shape.
2. **Maximum length stays 16,384 tokens** (decision 004); truncated stays incorrect.
3. **The selection halt checks format only:** stop before freezing if fewer than 80% of the
   selection samples that *stopped* contain a `\boxed{}` answer. This replaces decision 005 item 6,
   before any selection sample exists.
4. The sampler stops at the chat template's turn-end token as well as EOS: Qwen3.5 base lists only
   `<|endoftext|>` (248044) as EOS, while turns end with `<|im_end|>` (248046).

## Consequences
- The hard subset will lean toward problems the base model cannot finish within 16K tokens.
  RL that shortens responses can turn truncations into solved problems, a length effect that
  looks like sharpening or expansion. Truncation rate is reported per set and checkpoint, and
  transitions are split by whether the earlier model's failures were truncations.
- Evaluation cost is about twice decision 004's estimate. Mean length is about 9.4K tokens, not
  4K, and throughput is about 1,300 tokens/s per A40 (vLLM 0.24, one 800-sample call; about
  1,050 tokens/s in 64-sample calls, because of the slow tail). One full checkpoint, 51,568
  samples, is therefore about 485M tokens, roughly 100 A40-hours. Throughput tuning is an M0.7
  task and an input to G1.
- The selection run is 35,648 samples, about 335M tokens: roughly 36 hours on two A40s.

## For the report
"All models are prompted in thinking mode (an open `<think>` block); the choice did not change
base accuracy on a training-pool probe (−1.0 ± 1.6 points). At the 16,384-token limit about a
third of base samples are truncated and scored incorrect; we report truncation separately."

## Alternatives considered
- **Empty think block (template default):** equal accuracy, no flag to forget, but the base
  model then emits a stray `</think>` in 63% of samples and SFT traces sit after an empty block.
- **32K for evaluation and RL:** fewer truncations, but about 1.3–1.6× evaluation cost and
  RL rollouts that may not fit 40 GB A100s.
- **32K for evaluation, 16K for RL:** cheaper training, but truncation changes become harder to
  interpret and evaluation still costs more.
