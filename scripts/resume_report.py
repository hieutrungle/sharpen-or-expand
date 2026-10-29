"""M1.5 kill-and-resume report: compare an uninterrupted control run with a run killed after step 8.

    python scripts/resume_report.py --control runs/<control> --test runs/<test> \
        --notes <notes.json> --report docs/reports/m1_resume_test.json

`--notes` carries facts only Slurm knows (the successor's ReqNodeList and gres, the hardware-guard
refusal), collected outside the container. Image only (reads the optimizer state with torch).
Bitwise-equal metrics are not expected: vLLM batching is not deterministic (D12).
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.trainer.rl.checkpoints import export_dir, read_records, saved_steps
from soe.trainer.rl.rollout_tokens import cumulative_rollout_tokens
from soe.trainer.rl.segments import merged_metrics, read_events
from soe.utils.run import read_run_info

POST_RESUME = ("critic/score/mean", "actor/entropy", "actor/grad_norm")


def prompts(run_dir: Path, step: int) -> Counter[str]:
    path = run_dir / "generations" / "train" / f"{step}.jsonl"
    return Counter(json.loads(line)["input"] for line in path.read_text().splitlines())


def optimizer_step(run_dir: Path, step: int) -> float:
    actor = run_dir / "checkpoints" / f"global_step_{step}" / "actor"
    shard = next(actor.glob("optim_world_size_*_rank_0.pt"))
    state = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)["state"]
    return float(next(iter(state.values()))["step"])


def compare(control: Path, test: Path) -> dict[str, Any]:
    config = load_config(test / "config.yaml")
    assert config.budget is not None and config.budget.max_steps is not None
    assert config.rollout is not None and config.checkpoint is not None
    last = config.budget.max_steps
    per_step = config.rollout.prompts_per_step * config.rollout.group_size
    m_control, m_test = merged_metrics(control / "metrics"), merged_metrics(test / "metrics")
    events = read_events(test)
    starts = [e for e in events if e.event == "start"]
    checks: dict[str, Any] = {}

    expected_steps = list(range(0, last + 1))
    checks["steps_continuous"] = {
        "control": list(m_control) == expected_steps,
        "test": list(m_test) == expected_steps,
    }
    records = read_records(test)
    tokens = [r.rollout_tokens for r in records]
    recount = [
        cumulative_rollout_tokens(test / "rollout_tokens", r.step, per_step) for r in records
    ]
    checks["rollout_tokens_continuous"] = {
        "checkpoints": [(r.step, r.rollout_tokens) for r in records],
        "increasing": tokens == sorted(tokens) and len(set(tokens)) == len(tokens),
        "manifest_equals_recount": tokens == recount,
        "token_segments": sorted(p.name for p in (test / "rollout_tokens").glob("segment-*.jsonl")),
    }
    lr = {s: (m_control[s].get("actor/lr"), m_test[s].get("actor/lr")) for s in range(1, last + 1)}
    checks["lr_schedule_continues"] = {
        "equal_every_step": all(a == b for a, b in lr.values()),
        "per_step": lr,
    }
    checks["optimizer_step_continues"] = {
        "control": optimizer_step(control, last),
        "test": optimizer_step(test, last),
    }
    resumed_from = max(e.resumed_from_step or 0 for e in starts)
    checks["dataloader_restored"] = {
        str(s): prompts(control, s) == prompts(test, s) for s in range(1, last + 1)
    }
    first = resumed_from + 1
    post: dict[str, Any] = {}
    for key in POST_RESUME:
        series = [m_control[s][key] for s in range(1, last + 1)]
        spread = max(abs(b - a) for a, b in zip(series, series[1:], strict=False))
        diff = abs(m_test[first][key] - m_control[first][key])
        post[key] = {
            "control": m_control[first][key],
            "test": m_test[first][key],
            "difference": diff,
            "control_max_step_change": spread,
            "within": diff <= spread,
        }
    checks["first_step_after_resume"] = {"step": first, **post}
    checks["segments"] = {
        "events": [e.model_dump(mode="json") for e in events],
        "two_segments": len(starts) == 2
        and [e.resumed_from_step for e in starts] == [0, resumed_from],
        "same_node": len({e.node for e in starts}) == 1,
        "same_hardware": len({e.hardware for e in starts}) == 1,
    }
    data_root = Path(os.environ["SOE_DATA"])
    run_id = read_run_info(control).run_id
    every, keep = config.checkpoint.save_every_steps, config.checkpoint.keep_last
    saves = list(range(every, last + 1, every))
    checks["pruning_keeps_exports"] = {
        "shards_left": saved_steps(control / "checkpoints"),
        "expected_shards": saves[-keep:],
        "exports": {str(s): export_dir(data_root, run_id, s).exists() for s in saves},
    }
    return checks


class ResumeReportCli(BaseSettings):
    model_config = SettingsConfigDict(cli_prog_name="scripts/resume_report.py", cli_kebab_case=True)

    control: Path
    test: Path
    notes: Path | None = None
    report: Path

    def cli_cmd(self) -> None:
        result = {
            "control": str(self.control),
            "test": str(self.test),
            **compare(self.control, self.test),
        }
        if self.notes is not None:
            result["slurm_notes"] = json.loads(self.notes.read_text())
        self.report.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    CliApp.run(ResumeReportCli)
