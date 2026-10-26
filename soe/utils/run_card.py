"""Run card: a one-page markdown summary of a run's config, provenance and compute accounting."""

from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from soe.trainer.config.experiment import load_config
from soe.utils.run import read_efficiency, read_run_info


def render_run_card(run_dir: Path) -> str:
    info = read_run_info(run_dir)
    config = load_config(run_dir / "config.yaml")
    efficiency = read_efficiency(run_dir)
    per_million = efficiency.newly_solved_per_million_rollout_tokens()
    versions = ", ".join(f"{k} {v}" for k, v in info.library_versions.items() if v) or "none"
    lines = [
        f"# Run card: {info.run_id}",
        "",
        "## Experiment",
        f"- config: {config.name} (milestone {config.milestone}, lane {config.lane})",
        f"- algorithm: {config.algorithm}, training mode: {config.training_mode}",
        f"- model: {config.model.hf_id} @ {config.model.revision[:12]}",
        f"- init checkpoint: {config.init_checkpoint or 'none'}",
        f"- seed: {info.seed}",
        f"- development run: {'yes (excluded from conclusions)' if config.development else 'no'}",
        "",
        "## Provenance",
        f"- commit: {info.git_commit}{' (DIRTY TREE)' if info.git_dirty else ''}",
        f"- config sha256: {info.config_sha256[:16]}",
        f"- host: {info.hostname}, slurm job: {info.slurm_job_id or 'none'}",
        f"- GPUs: {'; '.join(info.gpus) or 'none detected'}",
        f"- libraries: {versions}",
        "",
        "## Compute accounting",
        f"- rollout tokens: {efficiency.rollout_tokens:,}",
        f"- training tokens: {efficiency.training_tokens:,}",
        f"- GPU-hours: {efficiency.gpu_hours:.1f}",
        f"- wall-clock hours: {efficiency.wall_clock_hours:.1f}",
        "- newly solved held-out problems per million rollout tokens: "
        + (f"{per_million:.3f}" if per_million is not None else "not measured yet"),
    ]
    return "\n".join(lines) + "\n"


def write_run_card(run_dir: Path) -> Path:
    path = run_dir / "card.md"
    path.write_text(render_run_card(run_dir))
    return path


class RunCardCli(BaseSettings):
    """Write runs/<run_id>/card.md."""

    model_config = SettingsConfigDict(
        cli_prog_name="python -m soe.utils.run_card", env_prefix="SOE_CARD_", cli_kebab_case=True
    )

    run_dir: Path

    def cli_cmd(self) -> None:
        print(write_run_card(self.run_dir))


if __name__ == "__main__":
    CliApp.run(RunCardCli)
