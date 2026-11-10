# 030 — The SFT checkpoint of record is run #2's final export

- Date: 2026-11-10
- Status: accepted (user, 2026-11-10). Item 1's role as the RL/DPO starting point is reopened by the
  base-start audit (`docs/reports/m2_base_start_audit.md`, 2026-11-10) and settled by decision 031:
  RL and DPO start from the base, and this export is an evaluated arm
- Milestone: M2 (task M2.11; decision 029 item 7)
- Reports: `docs/reports/m2_sft_run2_check.json` (run #1), `docs/reports/m2_sft_run2_check_run2.json`
  (run #2), `docs/reports/m2_sft_diagnostic.json` / `m2_sft_diagnostic_run2.json` (diagnostic only),
  `docs/reports/m2_sft_run2_export_check.json`

## Context

Run #1 (lr 1e-5, all 25,551 examples) met criteria 3 and 4 of decision 029, so run #2 applied both
pre-registered changes: lr 1e-5 / 3 and the 90th-percentile length cut (22,996 examples, 85.9M
tokens, `train_run2.jsonl` sha256 `c071eb6a…`). Run #2 (run dir
`runs/20261109-185607-sft_m2_run2_qwen35-4b-s1-j3237928`, seed 1, fc098, `baa5a13`) also meets
criteria 3 and 4. Decision 029 item 7 then allows no third run: this memo chooses between the two
final exports.

On the RL validation split (256 problems × 16, soe-eval-v4):

| | base | run #1 | run #2 |
|---|---|---|---|
| pass@1 | 0.660 | 0.608 | 0.637 |
| pass@16 | 0.852 | 0.836 | 0.855 |
| truncation | 0.321 | 0.367 | 0.345 |
| loops | 0.051 | 0.112 | 0.104 |
| accuracy when stopped | 0.972 | 0.960 | 0.972 |

Paired against the base (SE over 2,000 bootstrap resamples of the problems): run #2's pass@1 drop is
0.023 (SE 0.008; run #1 0.052), its truncation rise 0.024 (SE 0.008; run #1 0.046), its pass@16
change +0.004 (SE 0.012). Its whole pass@1 drop comes from stopping less (−0.023), none from
accuracy when stopped (+0.0003). The held-out DeepMath diagnostic (500 × 4, not part of the rule)
agrees: pass@1 0.868 / 0.830 / 0.849, truncation 0.078 / 0.122 / 0.105 for base / run #1 / run #2.

## Decision

1. **The frozen SFT checkpoint is run #2's final export**,
   `checkpoints/20261109-185607-sft_m2_run2_qwen35-4b-s1-j3237928/step_719/hf` under `$SOE_DATA`.
   M2.12 hashes it, copies it read-only and names it in `frozen_sft.yaml`.
2. **Why run #2:** it is no worse than run #1 on any measure of the rule: higher pass@1, lower
   truncation, pass@16 at the base's level, fewer loops. No choice between the two needs a
   trade-off, so the choice adds no selection beyond the mechanical check.
3. **Neither export meets decision 029's purpose for truncation.** SFT on the base model's own
   traces raises truncation (RL validation +2.4 points) and roughly doubles loops; it does not
   narrow coverage. This is reported, not repaired: decision 029 is not revised.

## Consequences (including what could confound the algorithm comparison)

- Every DPO and RL run starts from this one export; nothing here differs between algorithms.
- The RL starting point solves slightly less at pass@1 than the base and truncates more, at equal
  pass@16. M2.13 reports the base→SFT transitions on soe-eval-v4, so later sharpening or expansion
  is read against the base as well as against SFT.
- Truncation is present from the start of RL (34.5% on the RL validation split at 16,384 tokens).
  The milestone plan runs DAPO's overlong ablation only if truncation appears; this is evidence
  that it will, to be confirmed on the RL runs' own logs.
- Truncated rollouts get reward 0, so every algorithm is pushed to stop. A stopping gain could look
  like a capability gain in pass@1; the stopping/accuracy split of `scripts/sft_diagnostic.py`
  should be reported for RL checkpoints too.
- Run #2's data are 22,996 examples, below the plan's 25K–50K, and lack the longest 10% of traces
  (accepted in decision 029 item 6).
- Run #2 ran in two segments (steps 1–580 in job 3237928, 581–719 in 3254035, both at `baa5a13`,
  same node) after a disk-quota failure at the final save; step 581 reproduced the lost segment's
  loss and learning rate exactly.
- Likely cause, untested: the SFT traces were sampled at the suite's temperature 0.6, so SFT
  sharpens an already sharpened distribution, and evaluation at 0.6 compounds it (more loops,
  shorter stopped answers, unchanged accuracy when stopped; a 3× lower lr halved the effect).
  Not tested here because it would only matter for a third run.

## Alternatives considered

- **Run #1's export:** worse on every measure above.
- **Test the temperature cause first** (about 6 A40-h at temperature 1.0): informative, but it can
  only matter if decision 029 is revised for a third run.
- **Revise decision 029 for run #3** (e.g. traces re-sampled at temperature 1.0, 89–179 A40-h plus
  about 7.5 h on fc098): breaks the pre-registered rule and uses the RL validation split for
  selection a third time.
- **Start RL from the base model:** contradicts the experiment tree (one frozen SFT checkpoint).
