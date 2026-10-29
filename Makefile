# Thin wrappers around uv; every target works on a fresh clone after `make install`.
.PHONY: hooks install test test-container lint format typecheck check figures container stage clean

install:
	uv sync --locked

# Run ruff + mypy on every commit.
hooks:
	uv run pre-commit install

# CPU tests only (tests/**/test_*_on_cpu.py, as in CI); GPU tests run on FASTER in the container.
test:
	uv run pytest -q $$(find tests -name 'test_*_on_cpu.py' | sort)

# Every test, CPU and container (tests/**/test_*_in_container.py: torch, verl, vLLM), in the image
# on a FASTER CPU node. Run on the FASTER login node from the clone; the log is in logs/slurm/.
test-container:
	sbatch --time=01:00:00 --mem=32G --job-name=soe-tests slurm/cpu.sbatch python -m pytest -q tests

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

check: lint typecheck test

# One script per figure ID: soe/viz/fig_<ID>.py reads runs/ and writes figures/<ID>.{png,svg}.
figures:
	@scripts=$$(ls soe/viz/fig_*.py 2>/dev/null); \
	if [ -z "$$scripts" ]; then echo "no figure scripts yet"; fi; \
	for s in $$scripts; do echo "== $$s"; uv run python $$s || exit 1; done

# Build the image next to the repo (needs sudo + apptainer on your machine), then rsync the
# .sif to FASTER (docs/cluster.md). Temp files go to soe-data, not /tmp.
DATA ?= ../soe-data
SIF ?= $(DATA)/sif/soe.sif
container:
	mkdir -p $(DATA)/sif $(DATA)/cache/apptainer-tmp
	sudo APPTAINER_TMPDIR=$(abspath $(DATA))/cache/apptainer-tmp apptainer build --force $(SIF) container/soe.def
	@ls -lh $(SIF)

# Download models and datasets listed in scripts/stage_assets.yaml into $(DATA)/hf.
stage:
	HF_HOME=$(abspath $(DATA))/hf uv run scripts/stage_offline.py --assets scripts/stage_assets.yaml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
