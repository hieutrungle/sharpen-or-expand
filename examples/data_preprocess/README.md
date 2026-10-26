# data_preprocess/

Builds every problem file the study uses, offline, from snapshots staged by
`scripts/stage_offline.py`. Decisions 002 (evaluation sets) and 003 (RL pool, Level B).

```bash
export SOE_DATA=~/research/soe-data HF_HOME=$SOE_DATA/hf HF_HUB_OFFLINE=1
uv run --extra data python examples/data_preprocess/m0_datasets.py \
    --config examples/data_preprocess/m0_datasets.yaml
```

Writes `$SOE_DATA/datasets/soe/v1/{eval/<set>.jsonl, rl/train.jsonl, rl/level_b.jsonl,
build_report.json}` (one `soe.utils.dataset.problems.Problem` per line) and copies the report to
`docs/reports/m0_datasets_v1.json`. The report holds counts, dropped ids and sha256 per file but no
problem text, so it is committed; the problem files never are (CC-BY-NC-SA sources). The build is
deterministic: a rebuild must reproduce the committed sha256 values. Any change to the config or
the code that alters a file is a new `version` (v2, ...), never an edit of v1.
