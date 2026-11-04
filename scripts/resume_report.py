"""Kill-and-resume report: compare an uninterrupted control run with a run killed and resumed.

    python scripts/resume_report.py --control runs/<control> --test runs/<test> \
        --notes <notes.json> --report docs/reports/m1_resume_test.json

RL runs (M1.5: killed after step 8) and SFT runs (M2.7, decision 028: killed after its second save)
get their own checks, chosen by the config's algorithm.

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

from soe.trainer.config.experiment import Algorithm, load_config
from soe.trainer.rl.checkpoints import export_dir, read_records, saved_steps
from soe.trainer.rl.rollout_tokens import cumulative_rollout_tokens
from soe.trainer.rl.segments import merged_metrics, read_events
from soe.trainer.sft.resume import STATE_FILE, TRAINER_DIR, continuity, read_example_log
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
    every, keep = config.checkpoint.save_every_steps, config.checkpoint.keep_last
    saves = list(range(every, last + 1, every))
    checks["pruning_keeps_exports"] = {"expected_shards": saves[-keep:]}
    for name, run in (("control", control), ("test", test)):
        run_id = read_run_info(run).run_id
        checks["pruning_keeps_exports"][name] = {
            "shards_left": saved_steps(run / "checkpoints"),
            "exports": {str(s): export_dir(data_root, run_id, s).exists() for s in saves},
        }
    return checks


SFT_POST_RESUME = ("loss", "grad_norm")


def latest_examples(run_dir: Path) -> dict[tuple[int, int], list[int]]:
    """(step, rank) -> examples, a later segment's record of a step replacing an earlier one."""
    by_segment = read_example_log(run_dir)
    merged: dict[tuple[int, int], list[int]] = {}
    for segment in sorted(by_segment, key=lambda s: int(s) if s.isdigit() else -1):
        merged.update(by_segment[segment])
    return merged


def file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_checks(export: Path, base: Path) -> dict[str, Any]:
    """Vision tower and mtp bit-identical to the base, config and tokenizer files byte-identical."""
    from safetensors import safe_open

    def tensors(directory: Path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
        out = {}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path, "pt") as f:
                for name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                    if name.startswith(prefixes):
                        out[name] = f.get_tensor(name)
        return out

    frozen = ("model.visual.", "mtp.")
    ours, theirs = tensors(export, frozen), tensors(base, frozen)
    files = sorted(p.name for p in base.iterdir() if p.is_file() and p.suffix != ".safetensors")
    info = json.loads((export / "soe_export.json").read_text())
    return {
        "training": info["training"],
        "tensors_changed_vs_base": info["tensors_changed_vs_base"],
        "trained_by_prefix": info["trained_by_prefix"],
        "copied_from_base": len(info["copied_from_base"]),
        "vision_and_mtp_bit_identical": set(ours) == set(theirs)
        and all(torch.equal(ours[k], theirs[k]) for k in theirs),
        "vision_and_mtp_tensors": len(theirs),
        "other_files_byte_identical": {
            name: (export / name).exists() and file_sha(export / name) == file_sha(base / name)
            for name in files
        },
    }


def compare_sft(control: Path, test: Path) -> dict[str, Any]:
    config = load_config(test / "config.yaml")
    sft = config.sft
    assert sft is not None and sft.max_steps is not None and sft.save_every_steps is not None
    last, every = sft.max_steps, sft.save_every_steps
    m_control, m_test = merged_metrics(control / "metrics"), merged_metrics(test / "metrics")
    events = read_events(test)
    starts = [e for e in events if e.event == "start"]
    resumed_from = max(e.resumed_from_step or 0 for e in starts)
    steps = range(1, last + 1)
    checks: dict[str, Any] = {}
    checks["steps_continuous"] = {
        name: all(s in m and "loss" in m[s] for s in steps)
        for name, m in (("control", m_control), ("test", m_test))
    }
    checks["segments"] = {
        "events": [e.model_dump(mode="json") for e in events],
        "two_segments": len(starts) == 2
        and [e.resumed_from_step for e in starts] == [0, resumed_from],
        "resumed_from_second_save": resumed_from == 2 * every,
        "same_node": len({e.node for e in starts}) == 1,
        "same_hardware": len({e.hardware for e in starts}) == 1,
    }
    gaps = continuity(test)
    control_examples, test_examples = latest_examples(control), latest_examples(test)
    checks["example_order"] = {
        "redone_steps": gaps.redone_steps,
        "redone_steps_repeat_their_examples": not gaps.mismatched_steps,
        "equal_to_control_every_step_and_rank": control_examples == test_examples,
        "logged_steps": len({s for s, _ in test_examples}),
    }
    lr = {s: (m_control[s].get("learning_rate"), m_test[s].get("learning_rate")) for s in steps}
    checks["lr_schedule_continues"] = {
        "equal_every_step": all(a == b for a, b in lr.values()),
        "per_step": lr,
    }
    final = f"checkpoint-{last}"
    counts = {}
    for name, run in (("control", control), ("test", test)):
        ckpt = run / TRAINER_DIR / final
        scheduler = torch.load(ckpt / "scheduler.pt", map_location="cpu", weights_only=False)
        state = json.loads((ckpt / STATE_FILE).read_text())
        counts[name] = {
            "scheduler_steps": scheduler["last_epoch"],
            "global_step": state["global_step"],
        }
    checks["optimizer_step_continues"] = counts | {
        "equal": counts["control"] == counts["test"] and counts["test"]["global_step"] == last
    }
    post: dict[str, Any] = {}
    for key in SFT_POST_RESUME:
        series = [m_control[s][key] for s in steps]
        spread = max(abs(b - a) for a, b in zip(series, series[1:], strict=False))
        diffs = {
            s: abs(m_test[s][key] - m_control[s][key]) for s in range(resumed_from + 1, last + 1)
        }
        post[key] = {
            "max_difference_after_resume": max(diffs.values()),
            "control_max_step_change": spread,
            "within": max(diffs.values()) <= spread,
            "per_step_difference": diffs,
        }
    checks["after_resume"] = {"first_step": resumed_from + 1, **post}
    data_root = Path(os.environ["SOE_DATA"])
    from soe.trainer.main_rl import model_path

    base = model_path(config, data_root)
    checks["export"] = export_checks(export_dir(data_root, read_run_info(test).run_id, last), base)
    saves = list(range(every, last + 1, every))
    checks["pruning"] = {
        "expected": saves[-sft.keep_last :],
        **{
            name: sorted(
                int(p.name.split("-")[1]) for p in (run / TRAINER_DIR).glob("checkpoint-*")
            )
            for name, run in (("control", control), ("test", test))
        },
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
            **(
                compare_sft(self.control, self.test)
                if load_config(self.test / "config.yaml").algorithm is Algorithm.SFT
                else compare(self.control, self.test)
            ),
        }
        if self.notes is not None:
            result["slurm_notes"] = json.loads(self.notes.read_text())
        self.report.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    CliApp.run(ResumeReportCli)
