"""GPU-hours of this project's Slurm jobs, by job family and GPU type (compute accounting).

Runs on a FASTER login node with the system Python (3.6, no project packages):
    /usr/bin/python3 scripts/gpu_hours.py --since 2026-10-25 [--until 2026-10-29] > <report>.json
A job family is the job name without its GPU-type suffix (soe-eval-v3-a40 -> soe-eval-v3).
Local-variable annotations are never evaluated (PEP 526), so builtin generics are safe on 3.6.
GPU-hours = allocated GPUs x elapsed time, so queued or cancelled jobs count zero.
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict

GPU = re.compile(r"gres/gpu(?::(\w+))?=(\d+)")
SUFFIX = re.compile(r"-(a100|a40|a30|a10|t4)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", default="now")
    parser.add_argument("--prefix", default="soe")
    args = parser.parse_args()
    fmt = "--format=JobID,JobName,AllocTRES,ElapsedRaw"
    cmd = ["sacct", "-X", "-n", "-P", "-S", args.since, "-E", args.until, fmt]
    # universal_newlines: text= needs Python 3.7
    out = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout  # noqa: UP021
    jobs: defaultdict[str, int] = defaultdict(int)
    hours: defaultdict[str, float] = defaultdict(float)
    by_gpu: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for line in out.splitlines():
        _, name, tres, elapsed = line.split("|")
        match = GPU.search(tres)
        if not match or not name.startswith(args.prefix):
            continue
        family = SUFFIX.sub("", name)
        job_hours = int(match.group(2)) * int(elapsed or 0) / 3600
        jobs[family] += 1
        hours[family] += job_hours
        by_gpu[family][match.group(1) or "unknown"] += job_hours
    rows: dict[str, object] = {
        name: {
            "jobs": jobs[name],
            "gpu_hours": round(hours[name], 1),
            "by_gpu": {k: round(v, 1) for k, v in sorted(by_gpu[name].items())},
        }
        for name in sorted(hours, key=lambda n: -hours[n])
    }
    report = {
        "since": args.since,
        "until": args.until,
        "total_gpu_hours": round(sum(hours.values()), 1),
    }
    print(json.dumps({**report, "families": rows}, indent=1))


if __name__ == "__main__":
    main()
