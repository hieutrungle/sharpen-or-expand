"""Markdown tables for the M0 report chapter, built only from committed reports and the suite.

    python scripts/report_m0_tables.py > /tmp/m0_tables.md
Each table is printed under a `<!-- table: name -->` marker; docs/report/m0.md embeds them.
"""

import json
from pathlib import Path

import yaml

REPORTS = Path("docs/reports")


def load(name: str) -> dict:  # type: ignore[type-arg]
    data: dict = json.loads((REPORTS / name).read_text())  # type: ignore[type-arg]
    return data


def table(header: list[str], rows: list[list[object]]) -> str:
    def cell(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    base = load("qwen35-4b-base_soe-eval-v3_report.json")
    suite = yaml.safe_load(Path("soe/trainer/config/eval_suite.yaml").read_text())
    out: dict[str, str] = {}

    level_of = {s["name"]: s["level"] for s in suite["problem_sets"]}
    out["sets"] = table(
        ["Set", "Level", "Problems", "pass@1", "pass@16", "maj@16", "truncated"],
        [
            [
                name,
                level_of[name],
                int(r["problems"]),
                r["pass@1"],
                r["pass@16"],
                r["maj@16"],
                r["truncation_rate"],
            ]
            for name, r in sorted(
                base["main"]["per_set"].items(), key=lambda x: (level_of[x[0]], x[0])
            )
        ],
    )
    out["levels"] = table(
        [
            "Level",
            "Problems",
            "pass@1",
            "pass@4",
            "pass@8",
            "pass@16",
            "maj@16",
            "truncated",
            "mean tokens",
        ],
        [
            [
                lvl,
                int(r["problems"]),
                r["pass@1"],
                r["pass@4"],
                r["pass@8"],
                r["pass@16"],
                r["maj@16"],
                r["truncation_rate"],
                int(r["mean_completion_tokens"]),
            ]
            for lvl, r in base["main"]["per_level"].items()
        ],
    )
    hard = {"all": base["hard"]["all"], **base["hard"]["per_level"]}
    out["hard"] = table(
        ["Hard tier", "Problems", "pass@1", "pass@16", "pass@32", "pass@64", "maj@64", "truncated"],
        [
            [
                k,
                int(r["problems"]),
                r["pass@1"],
                r["pass@16"],
                r["pass@32"],
                r["pass@64"],
                r["maj@64"],
                r["truncation_rate"],
            ]
            for k, r in hard.items()
        ],
    )
    s = suite["sampling"]
    out["protocol"] = table(
        ["Setting", "Value", "Decision"],
        [
            [
                "temperature / top-p / top-k",
                f"{s['temperature']} / {s['top_p']} / {s['top_k']}",
                "004",
            ],
            ["max new tokens (also the RL rollout cap)", s["max_tokens"], "004, 006"],
            ["vLLM seed per request", s["seed"], "004"],
            ["KV cache", s["kv_cache_dtype"], "009"],
            [
                "chat template, thinking mode",
                f"{suite['prompt']['use_chat_template']}, {suite['prompt']['enable_thinking']}",
                "006",
            ],
            ["answer", "last \\boxed{}; else last 2-3 final-answer boxes joined", "004, 010"],
            ["truncated response", suite["scoring"]["truncated_response"], "004"],
            ["main tier k", ", ".join(map(str, suite["main"]["ks"])), "004"],
            ["hard tier k", ", ".join(map(str, suite["hard"]["ks"])), "005"],
        ],
    )
    fp8 = load("m0_fp8_vs_bf16.json")
    rows = []
    for group in ["level A", "level B", "level C1", "level C2", "hard tier"]:
        r = fp8[group]
        k = "pass@64" if group == "hard tier" else "pass@16"
        rows.append(
            [
                group,
                int(r["pass@1"]["problems"]),
                f"{100 * r['pass@1']['diff']:+.2f} ± {100 * r['pass@1']['se']:.2f}",
                f"{100 * r[k]['diff']:+.2f} ± {100 * r[k]['se']:.2f} ({k})",
            ]
        )
    ctl = fp8["bf16_seed_control"]
    for group in ["level A", "level B"]:
        r = ctl[group]
        rows.append(
            [
                f"control: BF16 seed 1 − seed 0, {group}",
                r["problems"],
                f"{100 * r['pass@1_diff']:+.2f} ± {100 * r['pass@1_se']:.2f}",
                f"{100 * r['pass@16_diff']:+.2f} ± {100 * r['pass@16_se']:.2f} (pass@16)",
            ]
        )
    out["fp8"] = table(["Comparison", "Problems", "pass@1 (points)", "pass@k (points)"], rows)

    v2, v3 = load("m0_datasets_v2.json"), load("m0_datasets_v3.json")
    out["data"] = table(
        ["File", "v2 (decision 007)", "v3 (decision 011)"],
        [
            ["rl/level_b (Level B)", v2["counts"]["rl/level_b"], "unchanged"],
            ["rl/train (RL pool)", v2["counts"]["rl/train"], v3["counts"]["rl/train"]],
            ["evaluation files", "11 sets, byte-identical to v1", "unchanged"],
        ],
    )
    c2, c3 = load("m0_contamination_rl.json"), load("m0_contamination_rl_v3.json")
    out["contamination"] = table(
        ["Check", "RL train v2", "RL train v3"],
        [
            [
                "residual at the frozen rule (13-grams >= 0.5)",
                len(c2["residual_at_frozen_rule"]["single_problem"]),
                len(c3["residual_at_frozen_rule"]["single_problem"]),
            ],
            *[
                [
                    f"option {name.split('_', 1)[0]}: {name.split('_', 1)[1]}",
                    v["train_dropped"],
                    c3["deletion_options"][name]["train_dropped"],
                ]
                for name, v in c2["deletion_options"].items()
            ],
        ],
    )
    excluded, removed = suite["excluded_problems"], suite["removed_from_hard_subset"]
    out["exclusions"] = table(
        ["List in the suite", "Problems"],
        [
            ["excluded_problems (wrong, ambiguous or unreadable gold)", len(excluded)],
            ["removed_from_hard_subset (verifier false negatives)", len(removed)],
            [
                "hard tier in the v3 report (frozen file of 250, minus both lists)",
                int(base["hard"]["all"]["problems"]),
            ],
        ],
    )
    comp = load("m0_compute.json")
    out["compute"] = table(
        ["Job family", "Jobs", "GPU-hours", "By GPU type"],
        [
            [k, v["jobs"], v["gpu_hours"], ", ".join(f"{g} {h}" for g, h in v["by_gpu"].items())]
            for k, v in comp["families"].items()
        ]
        + [["total", "", comp["total_gpu_hours"], ""]],
    )
    for name, text in out.items():
        print(f"<!-- table: {name} -->\n{text}\n")


if __name__ == "__main__":
    main()
