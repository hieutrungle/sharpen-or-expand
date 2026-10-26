# examples/

Every experiment is one YAML file validated by `soe.trainer.config.experiment.ExperimentConfig`;
nothing is hard-coded in Python. As in verl, files are grouped by algorithm:

```
examples/
  data_preprocess/     one script per dataset -> parquet under $SOE_DATA/datasets (from M0)
  grpo_trainer/        m1_dev_qwen35-2b.yaml, m3_screen_qwen35-4b.yaml, m5_full_qwen35-4b.yaml, ...
  dr_grpo_trainer/  dapo_trainer/  rs_grpo_trainer/  sft/  dpo/      (added as milestones need them)
```

- File name: `m<k>_<purpose>_<model>.yaml`. The `name` field is `<algorithm>_<file stem>`
  (e.g. `grpo_m1_dev_qwen35-2b`) and is what run directories are named after.
- Leave-one-out ablations are separate files that differ from the full recipe in one field;
  `git diff --no-index` between them should show exactly that field.
- The frozen evaluation suite is `soe/trainer/config/eval_suite.yaml` (created in M0). It is never
  edited after the freeze; a change needs a decision note and a new suite name.
