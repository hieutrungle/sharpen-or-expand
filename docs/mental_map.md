The export never runs inside verl. verl only triggers it, through a small callback after each checkpoint save. The export is a separate CPU process that turns the saved files into an HF model directory, while training keeps going on the GPUs.

## Who runs where

```
Slurm job (train.sbatch) on the GPU node
│
└─ main_rl  (soe launcher, one per job segment)
   │  sets env: SOE_RUN_DIR, SOE_ROLLOUT_TOKENS, SOE_ROLLOUT_FAILURES, VERL_FILE_LOGGER_PATH
   │
   ├─ [start] sweep: export_missing()  ──► main_export (blocking, one per missing step)
   │
   ├─ verl.trainer.main_ppo  (subprocess; Ray driver + GPU workers)
   │    │
   │    ├─ agent loop (each rollout) ──► appends {step, tokens} to rollout_tokens/segment-<job>.jsonl
   │    │
   │    └─ every save_every_steps:
   │         verl writes checkpoints/global_step_N/  (FSDP shards, optimizer, data.pt)
   │         verl calls SoeCheckpointCallback.on_save(N)
   │              ├─ sum tokens for steps 1..N (fails the run if a rollout is missing)
   │              ├─ append one line to checkpoints.jsonl
   │              └─ start main_export --step N in the background ──► (training continues)
   │
   └─ [end, exit 0] sweep again ──► main_export for any step still missing; then write DONE
```

## What `main_export` does

```
global_step_N/actor/*.pt  ─┐
lora_train_meta.json      ─┼─► 1 join shards (FP32)
optim rank-0 shard        ─┘   2 which params were trained? (only language model allowed)
                               3 merge LoRA:  W = W0 + (alpha/r)·B·A
base HF snapshot ────────────► 4 compare with base, tensor by tensor:
                                  - language model: may differ
                                  - vision tower: must be identical
                                  - mtp.*: copied from base
                                  - tied lm_head: dropped
                               5 write .hf.tmp-<pid>/ in base dtypes and layout + soe_export.json
                               6 rename → $SOE_DATA/checkpoints/<run_id>/step_N/hf
                                                     │
                                       Lane B: main_eval --checkpoint <that dir>
```

## Three ideas that hold it together

1. **Training and evaluation only meet through files.** Lane A writes checkpoints and exports; Lane B reads the finished `step_N/hf` directory. No job waits on the other.
2. **An export either exists completely or not at all.** It is built in a temporary directory and renamed at the end. "Is step N exported?" is simply "does `step_N/hf` exist?", which is how both the sweep and `main_export` decide to skip.
3. **Three paths lead to the same export, and duplicates are harmless.**
   - the callback, right after each save (normal case);
   - the sweep at job start, for a previous job that died mid-export;
   - the sweep at job end, for the last save.

   If two run at once for the same step, the second finds the directory already there and discards its own copy.

## The files involved

| File | Writer | Reader |
|---|---|---|
| `runs/<id>/rollout_tokens/segment-<job>.jsonl` | agent loop | callback (token sum) |
| `runs/<id>/checkpoints/global_step_N/` | verl | `main_export`; verl on resume |
| `runs/<id>/checkpoints.jsonl` | callback | `main_export` (token count), later comparisons at equal tokens |
| `runs/<id>/exports/step_N.log` | background `main_export` | you, when debugging |
| `$SOE_DATA/checkpoints/<run_id>/step_N/hf/` | `main_export` | `main_eval` on Lane B |

One timing fact explains why the export runs inside the training job. verl deletes the old step's shards during the next save, 2–4 hours later, and an export takes 1–2 minutes. Started right after the save, it finishes long before its input disappears.