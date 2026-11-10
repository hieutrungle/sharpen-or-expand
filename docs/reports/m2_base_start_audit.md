# Pre-switch audit: RL and DPO from Qwen3.5-4B-Base instead of SFT run #2

Date 2026-11-10. Base = `Qwen/Qwen3.5-4B-Base` @ `1001bb4d826a52d1f399e183466143f4da7b741b`.
Asked by the user before deciding the RL starting point; it chooses nothing. Measured at `d128bdc` by
`scripts/format_audit.py` → `m2_base_format_audit.json` (job 3255478), `scripts/prompt_identity.py` →
`m2_prompt_identity.json` (job 3255479, inside `soe.sif`, verl 0.9.1) and `scripts/rollout_groups.py`
→ `m2_rollout_groups.json` (job 3255480, the M1 4B throughput rollout dumps); the rest read from the
repo and verl's source.

## A. Checks that pass

| # | Check | Evidence | Result |
|---|---|---|---|
| A1 | Prompt identity: verl rollout prompt ids = evaluation prompt ids | verl's own `QwenVLContinuousTokenBuilder` (processor template synced as `HFModelConfig` does) vs the eval renderer, every RL problem | 256/256 val, 11,868/11,868 train identical; all end `<\|im_start\|>assistant\n<think>\n`; user text = suite prompt for all |
| A2 | Demonstration format at eval sampling (0.6/0.95) | format audit, base RL val 4,096 + held-out 2,000 samples vs 22,996 demos | one `</think>` in 100% of stopped samples; box after it 99.75% (demos 99.85%); similar answer length; no leaked turns, CJK or special text |
| A3 | Format at rollout sampling (1.0/1.0), real verl path | M1 4B rollout dumps, untrained base at step 1 (512 rollouts) and steps 1–14 (7,168) | when stopped: box 100% at step 1 (99.98% pooled), ends on `<\|im_end\|>` 100% (99.98% LoRA pooled) |
| A4 | Stop / EOS tokens | config.json EOS 248044; eval sampler and agent loop stop on [248044, 248046] | base ends turns with `<\|im_end\|>` (A3); truncation flag = no stop token |
| A5 | Hidden sampling defaults | base snapshot has no `generation_config.json` (nor does the SFT export) | none; temperature/top-p/top-k explicit in both paths |
| A6 | Tokenizer / config files | SFT export check: tokenizer and config files byte-identical to base | same rendering for either start |
| A7 | Reference model / KL | decision 013: no KL, no reference model | start checkpoint has no second role |
| A8 | Hard subset selection bias | decision 005: selected by 4B-Base with vLLM seed 1; evaluation uses seed 0 | base's hard-subset rates are fresh draws, unbiased |
| A9 | Base baseline exists | `qwen35-4b-base_soe-eval-v4_report.json`: Levels A, B, C1, C2 at K up to 16; hard subset (233) to K = 64 | complete; nothing to re-run |
| A10 | RL hyperparameters (D8) | lr 1e-6, 10 warmup steps, wd 0.1 from DAPO, which trains from a base model | consistent with a base start |
| A11 | RL machinery on 4B base | M1 memory/throughput runs (full BF16 and LoRA) trained from the base at 1.0/1.0 | ran; reward 0.61 → 0.75 over 14 steps |

## B. Findings to carry (not blockers)

1. **Learning signal.** Base at G = 8 on the RL train set (896 groups): 44% all-correct, 16% all-wrong,
   40% with signal; step 1 alone 41% / 22% / 37%. The same RL set is used from either start.
2. **Early RL from base is mostly learning to stop.** Truncation 36% at step 1 (29% pooled); accuracy when
   stopped 95%; 90% of rollouts in all-wrong groups are truncated. RL pass@1 gains must be split into
   stopping vs accuracy (`scripts/sft_diagnostic.py` method) and read with truncation, for every algorithm.
3. **Hard subset truncation 90%** for the base (pass@64 0.489): its gains will also largely be stopping gains.
4. **Gemma 4 E4B base format is unknown.** Under B its replication starts from its base; a small format
   probe with Gemma's template is needed before M8 so the two models stay symmetric.
5. Evidence A3/A11 comes from M1-era runs (placeholder lr; steps ≥ 2 slightly trained); M2.14's 4B smoke
   from base on the frozen image is the end-to-end confirmation.

## C. What option B changes (to draft for review; nothing changed yet)

- `soe/trainer/config/experiment.py`: the validator requiring `init_checkpoint` for non-development RL/DPO
  runs (lines 251–254) becomes "start = pinned base, `init_checkpoint` null"; tests updated.
- New in-container test: verl prompt ids = eval prompt ids (A1), so a library or template change cannot
  break the base's format silently.
- Decisions: 030 rewritten (start = base; SFT run #2 kept as an evaluated arm); 025 "Start" row revised;
  014/017 "DPO pairs from the SFT policy" → base policy; 029 consequence note.
- Plan and roadmap: experiment tree, research rule "every run starts from the frozen SFT checkpoint", M2
  exit criteria, M2.12–M2.14, M4 DPO, M8 Gemma; ROADMAP.md / README.md.
- Outside the repo (yours to edit): the project instructions' research rule and the milestone-plan artifact.
- M2.12 becomes freezing the start: record the base snapshot's file sha256s in a frozen-start file that
  every config must name; M2.13 evaluates SFT run #2 on the full suite as an arm.
