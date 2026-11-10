# Roadmap

Milestones and gates from [docs/milestone-plan.md](docs/milestone-plan.md). A milestone is done
only when every exit box in the plan is ticked. Each milestone gets a detailed
`docs/roadmap/Mx.md` (from [TEMPLATE.md](docs/roadmap/TEMPLATE.md)) when the previous one closes.

| Milestone | Weeks | Lane | Content | Gate | Status |
| --- | --- | --- | --- | --- | --- |
| [M0](docs/roadmap/M0.md) Frozen evaluation and verifier | 1 | B | Level A/B/C, hard subset, sampling params, verifier audit, contamination report; repo + offline infra | | done (2026-10-28) |
| [M1](docs/roadmap/M1.md) Working pipeline and 4B memory study | 1–2 | A, B | 2B SFT/GRPO pipeline, kill-and-resume, 4B full BF16 vs LoRA at G=8 | **G0** training mode ([memo](docs/gates/G0.md)) | done (2026-11-03) |
| [M2](docs/roadmap/M2.md) Frozen start (base, decision 031) and the SFT arm | 2–3 | A, B | 25K–50K verified trajectories, SFT, base/SFT eval | | in progress |
| M3 RL screening | 4–5 | A, B | GRPO, Dr.GRPO, DAPO, RS-GRPO at 30–50M rollout tokens; preferences; β sweep on 2B | **G1** budget + stopping rule | |
| M4 DPO | 6 | A | DPO from the frozen start (the base) | | |
| M5 Full seed-1 runs | 6–8 | A | GRPO, DAPO, RS-GRPO to the G1 budget | **G2** go/no-go, hypothesis locked | |
| M6 Seeds 2 and 3 | 9–11 | A | six more full runs | **G3** do seeds confirm it | |
| M7 Ablations | 12–13 | A | DAPO leave-one-out; GRPO G = 4, 8, 16 | | |
| M8 Gemma 4 E4B replication | 14–15 | A | Gemma SFT + the two implicated methods | | |
| M9 Final evaluation and write-up | 16 | B | gaps, compute accounting, report | | |
