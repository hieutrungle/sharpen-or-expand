# 012 — In-loop validation split carved from RL train v3; RL data v3.1

- Date: 2026-10-28
- Status: accepted (user, 2026-10-28: D1 option (a), "256 problems carved from RL train v3")
- Milestone: M1 (task M1.2; amends the plan's "One yardstick" rule: RL train v3 → v3.1)
- Built by: `examples/data_preprocess/m1_rl_data.py` + `m1_rl_data.yaml`; report
  `docs/reports/m1_datasets_v3.1.json`

## Context

RL jobs run a small in-loop pass@1 validation, which later becomes the G1 stopping signal. The plan
did not say which problems it uses. Level B would be the obvious choice, but stopping on Level B
selects each run on the set that then measures it. Noisier runs gain more from that selection, and
Level B transitions are a headline measurement. Level C is never used for any selection.

## Decision

1. **Validation = 256 problems carved from RL train v3** (12,124 problems, decision 011) by
   `random.Random(20261029)`.
2. **Eligibility, no extra drops.** A problem can enter validation only if the rest of train covers
   less than half of its word 13-grams. This is decision 003's union rule (`covered_by_union`),
   applied as an eligibility test, with 13-grams found in more than 20 problems ignored as
   boilerplate (48 n-grams). Train loses the 256 validation problems and nothing else, following the
   project rule of not dropping training data that is not a copy.
3. **Result: RL data v3.1** in `$SOE_DATA/datasets/soe/v3.1/rl/`:
   - `train.jsonl`: 11,868 problems; `val.jsonl`: 256 problems;
   - both as verl parquet (`verl/{train,val}.parquet`) with the frozen suite's prompt rendered in;
   - 10,854 problems were eligible. The largest shared fraction in validation is 0.48;
   - audit: no validation problem is covered by train, and no train problem near-duplicates a
     validation problem;
   - the build is deterministic (byte-identical rebuild, pyarrow 26.0.0).
4. Every RL run and every preference set uses train v3.1; validation problems are never trained on.
   The evaluation suite (soe-eval-v3), Level B and the hard subset are unchanged.

## Consequences (including what could confound the algorithm comparison)

- Every algorithm validates and stops on the same 256 problems with the same sampling (the suite's:
  temperature 0.6, top-p 0.95, 16K, seed 0), so the choice cannot favour one algorithm.
- The 1,270 ineligible problems belong to templated families (variants of each other). Validation
  therefore leans toward problems without close variants in train, which is the point: it measures
  generalisation within the distribution, not recall of a sibling problem.
- 256 problems give a pass@1 standard error of about ±3 points at 50% accuracy per sample. Its
  sampling cost is decided at G1 (`validation.max_problems`, `samples_per_problem`); M1 measures it.
- 2% less training data than v3.

## Alternatives considered

- **(b) A fixed 256-problem Level B subset:** no new data version, but stopping selects on the
  measured set.
- **(c) A Level B subset dropped from Level B reporting:** clean, but changes what "Level B" means in
  every report and shrinks it.
- **Draw from all of train v3 and drop train problems that near-duplicate the drawn ones:** removes
  training data beyond the 256.
