"""M2.11 format audit: does a model's output follow the SFT demonstration format, part by part?

    E=$SOE_DATA/eval; S=$SOE_DATA/datasets/soe/v1/sft; python scripts/format_audit.py \\
        --demos demos_train_run2:$S/train_run2.jsonl \\
        --eval base_rl_val:$E/m2.11-base-rl-val --eval run2_rl_val:$E/m2.11-sft-run2-rl-val \\
        --eval base_heldout:$E/m2.11diag-base-sft-val \\
        --eval run2_heldout:$E/m2.11diag-sft-run2-sft-val \\
        --report docs/reports/m2_base_format_audit.json

`--eval` is `name:main_eval output dir`; `--demos` is `name:SFT train file` (every demonstration
counts as stopped). For stopped samples: how many `</think>` (exactly one expected), a reopened
`<think>`, boxed answers after `</think>` (at least one, exactly one, several distinct), answer
section and thinking lengths, characters after the last box. For all samples: a chat turn leaked
into the text, CJK characters, special-token text. For truncated samples: whether the cut fell in
the thinking or the answer, and the loop heuristic of scripts/sft_diagnostic.py. Asked by the
user (2026-11-10) before deciding the RL starting point; it chooses nothing. Numbers only.
"""

import json
import re
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.utils.dataset.records import SampleRecord
from soe.utils.jsonl import read_jsonl

TAIL_CHARS, LOOP_RATIO = 4000, 0.15  # as scripts/sft_diagnostic.py
THINK_CLOSE, THINK_OPEN, BOX = "</think>", "<think>", "\\boxed{"
SPECIAL_TEXT = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
LEAKED_TURN = re.compile(r"\n(user|assistant|system|Human|User|Assistant)\s*:?\s*\n")
CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    completion: str
    stopped: bool


def boxed_contents(text: str) -> list[tuple[int, str]]:
    """(end offset, content) of every `\\boxed{...}`, braces matched."""
    out: list[tuple[int, str]] = []
    start = text.find(BOX)
    while start != -1:
        end, depth = start + len(BOX), 1
        while end < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        out.append((end, text[start + len(BOX) : end - 1]))
        start = text.find(BOX, end)
    return out


def loops(text: str) -> bool:
    tail = text[-TAIL_CHARS:].encode()
    return len(zlib.compress(tail)) / max(1, len(tail)) < LOOP_RATIO


def quantile(values: list[int], q: float) -> int | None:
    return sorted(values)[int(q * (len(values) - 1))] if values else None


def audit(samples: Iterable[Sample]) -> dict[str, Any]:
    c: Counter[str] = Counter()
    answer_chars: list[int] = []
    think_chars: list[int] = []
    after_box: list[int] = []
    for s in samples:
        text = s.completion
        c["samples"] += 1
        c["leaked_turn"] += bool(LEAKED_TURN.search(text))
        c["cjk"] += bool(CJK.search(text))
        c["special_text"] += any(t in text for t in SPECIAL_TEXT)
        closes = text.count(THINK_CLOSE)
        if not s.stopped:
            c["truncated"] += 1
            c["cut_in_thinking"] += closes == 0
            c["cut_in_answer"] += closes >= 1
            c["cut_loop"] += loops(text)
            continue
        c["stopped"] += 1
        c["one_close"] += closes == 1
        c["reopened_think"] += THINK_OPEN in text
        if closes != 1:
            continue
        think, answer = text.split(THINK_CLOSE, 1)
        boxes = boxed_contents(answer)
        c["box_ge1"] += len(boxes) >= 1
        c["box_eq1"] += len(boxes) == 1
        c["boxes_distinct_gt1"] += len({b.strip() for _, b in boxes}) > 1
        answer_chars.append(len(answer))
        think_chars.append(len(think))
        if boxes:
            after_box.append(len(answer) - boxes[-1][0])

    def share(key: str, of: str) -> float | None:
        return c[key] / c[of] if c[of] else None

    return {
        "samples": c["samples"],
        "stop_rate": share("stopped", "samples"),
        "when_stopped": {k: share(k, "stopped") for k in ("one_close", "reopened_think")},
        "when_one_close": {
            k: share(k, "one_close") for k in ("box_ge1", "box_eq1", "boxes_distinct_gt1")
        },
        "answer_chars_p10_p50_p90": [quantile(answer_chars, q) for q in (0.1, 0.5, 0.9)],
        "think_chars_p50": quantile(think_chars, 0.5),
        "chars_after_last_box_p50_p90": [quantile(after_box, q) for q in (0.5, 0.9)],
        "all": {k: share(k, "samples") for k in ("leaked_turn", "cjk", "special_text")},
        "when_truncated": {
            k: share(k, "truncated") for k in ("cut_in_thinking", "cut_in_answer", "cut_loop")
        },
    }


def eval_samples(directory: Path) -> list[Sample]:
    plan = json.loads((directory / "plan.json").read_text())
    files = sorted(directory.glob(f"samples-*-of-{plan['num_shards']}.jsonl"))
    if len(files) != plan["num_shards"]:
        raise SystemExit(f"{directory}: {len(files)} of {plan['num_shards']} shards")
    records = [r for f in files for r in read_jsonl(f, SampleRecord)]
    return [Sample(completion=r.completion, stopped=r.finish_reason == "stop") for r in records]


def demo_samples(path: Path) -> list[Sample]:
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return [Sample(completion=r["completion"], stopped=True) for r in rows]


class FormatAuditCli(BaseSettings):
    model_config = SettingsConfigDict(cli_prog_name="scripts/format_audit.py", cli_kebab_case=True)

    eval: list[str] = []  # "name:main_eval output dir"
    demos: list[str] = []  # "name:SFT train jsonl"
    report: Path

    def cli_cmd(self) -> None:
        sets: dict[str, list[Sample]] = {}
        for item in self.demos:
            name, path = item.split(":", 1)
            sets[name] = demo_samples(Path(path))
        for item in self.eval:
            name, path = item.split(":", 1)
            sets[name] = eval_samples(Path(path))
        report = {name: audit(samples) for name, samples in sets.items()}
        self.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        for name, r in report.items():
            print(name, json.dumps({k: r[k] for k in ("stop_rate", "when_one_close")}))


if __name__ == "__main__":
    CliApp.run(FormatAuditCli)
