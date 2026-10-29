from datetime import UTC, datetime
from pathlib import Path

from soe.trainer.rl.checkpoints import (
    CheckpointRecord,
    append_record,
    export_dir,
    hf_name,
    is_language_model,
    lora_parts,
    missing_exports,
    prune_shards,
    read_records,
    saved_steps,
)

# Key names as they appear in the M1.3 smoke checkpoint (verl 0.9.1, PEFT, FSDP2).
LORA_A = "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.default.weight"
BASE_LAYER = "base_model.model.model.language_model.layers.0.linear_attn.out_proj.base_layer.weight"
VISUAL = "base_model.model.model.visual.patch_embed.proj.bias"


def test_fsdp_keys_map_to_hf_names() -> None:
    assert hf_name(BASE_LAYER) == "model.language_model.layers.0.linear_attn.out_proj.weight"
    assert hf_name(VISUAL) == "model.visual.patch_embed.proj.bias"
    assert hf_name("model.language_model.norm.weight") == "model.language_model.norm.weight"  # full
    assert lora_parts(LORA_A) == ("model.language_model.layers.0.linear_attn.out_proj.weight", "A")
    assert lora_parts(BASE_LAYER) is None
    assert is_language_model(hf_name(BASE_LAYER)) and not is_language_model(hf_name(VISUAL))
    assert not is_language_model("mtp.fc.weight")


def record(step: int, tokens: int) -> CheckpointRecord:
    return CheckpointRecord(
        step=step,
        rollout_tokens=tokens,
        saved_at=datetime(2026, 10, 28, tzinfo=UTC),
        slurm_job_id="1",
        hardware="NVIDIA A40",
        checkpoint_dir=f"/r/checkpoints/global_step_{step}",
        export_dir=f"/d/step_{step}/hf",
    )


def test_manifest_keeps_the_latest_entry_per_step(tmp_path: Path) -> None:
    append_record(tmp_path, record(2, 100))
    append_record(tmp_path, record(4, 250))
    append_record(tmp_path, record(4, 260))  # step 4 saved again after a resume
    assert [(r.step, r.rollout_tokens) for r in read_records(tmp_path)] == [(2, 100), (4, 260)]


def test_missing_exports_lists_saved_steps_without_an_export(tmp_path: Path) -> None:
    checkpoints, data = tmp_path / "checkpoints", tmp_path / "data"
    for step in (2, 4):
        actor = checkpoints / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        (actor / "model_world_size_2_rank_0.pt").write_bytes(b"")
    (checkpoints / "global_step_1").mkdir()  # pruned: data.pt only, no actor shards
    export_dir(data, "run", 2).mkdir(parents=True)
    assert missing_exports(checkpoints, data, "run") == [4]


def test_prune_shards_removes_older_shards_saved_by_an_earlier_job(tmp_path: Path) -> None:
    # The M1.5 kill test: steps 3 and 6 saved by the killed job, 9 and 12 after the resume.
    checkpoints, data = tmp_path / "checkpoints", tmp_path / "data"
    for step in (3, 6, 9, 12):
        actor = checkpoints / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        (actor / "model_world_size_2_rank_0.pt").write_bytes(b"")
        (checkpoints / f"global_step_{step}" / "data.pt").write_bytes(b"")
    for step in (3, 9, 12):
        export_dir(data, "run", step).mkdir(parents=True)
    assert prune_shards(checkpoints, data, "run", keep_last=2) == [3]
    assert saved_steps(checkpoints) == [6, 9, 12]  # step 6 waits for its export
    export_dir(data, "run", 6).mkdir(parents=True)
    assert prune_shards(checkpoints, data, "run", keep_last=2) == [6]
    assert saved_steps(checkpoints) == [9, 12]
    assert (checkpoints / "global_step_3" / "data.pt").exists()
