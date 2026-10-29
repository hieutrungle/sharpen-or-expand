"""Reward fixtures for M1.7: 200 scored RL rollouts drawn from verl's validation dumps.

    python scripts/make_reward_fixtures.py --archive $SOE_DATA/archive/m1.5_resume_runs.tgz \
        --out tests/workers/reward_manager/fixtures/reward_fixtures.jsonl.gz

The validation split is carved from the RL-train pool (decision 012), so no evaluation text enters
git. The dumps (`generations/val/<step>.jsonl` in each run) hold the decoded response, the gold
answer, the training-time reward and the reward hook's `truncated`/`has_box` flags. Rows are drawn
per category (correct, wrong with a box, no box, truncated) with a fixed seed. Rows that stopped
keep their whole text, so only short ones are drawn; a truncated row keeps the tail only, since it
gets reward 0 whatever it says (decision 014).
"""

import gzip
import json
import random
import tarfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

PER_CATEGORY = {"correct": 70, "wrong_boxed": 70, "no_box": 30, "truncated": 30}
MAX_CHARS = 6000  # stopped rows: whole text up to this length
TAIL_CHARS = 2000  # truncated rows
SEED = 20261029


class RewardFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    source: str  # archive member and line
    response: str
    gold: str
    truncated: bool
    training_reward: float  # what the reward hook returned during the run


def category(row: dict[str, object]) -> str:
    if row["truncated"]:
        return "truncated"
    if float(str(row["score"])) == 1.0:
        return "correct"
    return "wrong_boxed" if row["has_box"] else "no_box"


def read_dumps(archive: Path) -> list[RewardFixture]:
    rows = []
    with tarfile.open(archive) as tar:
        members = sorted(
            (m for m in tar.getmembers() if "/generations/val/" in m.name and m.isfile()),
            key=lambda m: m.name,
        )
        for member in members:
            handle = tar.extractfile(member)
            assert handle is not None
            for i, line in enumerate(handle.read().decode().splitlines()):
                row = json.loads(line)
                kind = category(row)
                text = str(row["output"])
                if kind == "truncated":
                    text = text[-TAIL_CHARS:]
                elif len(text) > MAX_CHARS:
                    continue
                rows.append(
                    RewardFixture(
                        category=kind,
                        source=f"{member.name}:{i}",
                        response=text,
                        gold=str(row["gts"]),
                        truncated=bool(row["truncated"]),
                        training_reward=float(str(row["score"])),
                    )
                )
    return rows


class FixturesCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="scripts/make_reward_fixtures.py", cli_kebab_case=True
    )

    archive: Path
    out: Path

    def cli_cmd(self) -> None:
        rows = read_dumps(self.archive)
        rng = random.Random(SEED)
        chosen: list[RewardFixture] = []
        for kind, n in PER_CATEGORY.items():
            pool = [r for r in rows if r.category == kind]
            if len(pool) < n:
                raise ValueError(f"only {len(pool)} {kind} rows, need {n}")
            chosen += rng.sample(pool, n)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(row.model_dump_json() + "\n" for row in chosen)
        # mtime=0: the same rows always give the same bytes.
        self.out.write_bytes(gzip.compress(text.encode(), mtime=0))
        counts = {k: sum(r.category == k for r in rows) for k in PER_CATEGORY}
        print(f"{len(chosen)} fixtures from {len(rows)} eligible rows {counts} -> {self.out}")


if __name__ == "__main__":
    CliApp.run(FixturesCli)
