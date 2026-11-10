"""M2.11 base-start audit: format and learning signal of RL rollouts from a run's rollout dump.

    A=runs/20261102-153916-m1_throughput_bf0_qwen35-4b-s20261029-j3230133
    B=runs/20261102-224210-m1_throughput_bl_qwen35-4b-s20261029-j3230134
    python scripts/rollout_groups.py \\
        --rollouts full_bf16:$A/rollouts/segment-3230133.jsonl \\
        --rollouts lora_bf16:$B/rollouts/segment-3230134.jsonl \\
        --report docs/reports/m2_rollout_groups.json

`--rollouts` is `name:rollout jsonl` (one line per rollout: step, problem_id, score, truncated,
has_box, turn_end). For step 1 (the untrained start) and for all steps: truncation, the box and
`<|im_end|>` shares among stopped rollouts, mean score when stopped, and the share of groups (one
problem in one step) that are all correct, all wrong, or mixed (the only ones with a GRPO
advantage), plus the truncated share inside all-wrong and mixed groups. Chooses nothing.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict


@dataclass(frozen=True, slots=True, kw_only=True)
class Rollout:
    step: int
    problem_id: str
    score: float
    truncated: bool
    has_box: bool
    turn_end: bool


def read_rollouts(path: Path) -> list[Rollout]:
    keys = ("step", "problem_id", "score", "truncated", "has_box", "turn_end")
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return [Rollout(**{k: r[k] for k in keys}) for r in rows]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summary(rollouts: list[Rollout]) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[Rollout]] = defaultdict(list)
    for r in rollouts:
        groups[(r.step, r.problem_id)].append(r)
    kind = {key: mean([r.score for r in g]) for key, g in groups.items()}
    stopped = [r for r in rollouts if not r.truncated]

    def truncated_in(selected: list[tuple[int, str]]) -> float | None:
        return mean([float(r.truncated) for key in selected for r in groups[key]])

    all_wrong = [k for k, m in kind.items() if m == 0]
    mixed = [k for k, m in kind.items() if m is not None and 0 < m < 1]
    return {
        "rollouts": len(rollouts),
        "groups": len(groups),
        "group_sizes": sorted({len(g) for g in groups.values()}),
        "truncated": mean([float(r.truncated) for r in rollouts]),
        "box_when_stopped": mean([float(r.has_box) for r in stopped]),
        "im_end_when_stopped": mean([float(r.turn_end) for r in stopped]),
        "score_when_stopped": mean([r.score for r in stopped]),
        "groups_all_correct": mean([float(m == 1) for m in kind.values() if m is not None]),
        "groups_all_wrong": len(all_wrong) / len(groups),
        "groups_mixed": len(mixed) / len(groups),
        "truncated_in_all_wrong_groups": truncated_in(all_wrong),
        "truncated_in_mixed_groups": truncated_in(mixed),
    }


class RolloutGroupsCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/rollout_groups.py", cli_kebab_case=True
    )

    rollouts: list[str]  # "name:rollout jsonl"
    report: Path

    def cli_cmd(self) -> None:
        report: dict[str, Any] = {}
        for item in self.rollouts:
            name, path = item.split(":", 1)
            rollouts = read_rollouts(Path(path))
            report[name] = {
                "file": path,
                "step_1": summary([r for r in rollouts if r.step == 1]),
                "all_steps": summary(rollouts),
            }
        self.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    CliApp.run(RolloutGroupsCli)
