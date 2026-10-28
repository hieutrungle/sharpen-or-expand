"""Compare an HF export with its base snapshot, tensor by tensor (M1.4, decision 018).

    python scripts/check_export.py --export <step_N/hf> --base <snapshot> --label <name> \
        --report docs/reports/m1_export_check.json

A step-0 identity export must have no differing tensor; a trained export may differ only inside
the language model. The result is stored under `label` in the report (other labels are kept).
Image only (torch, safetensors).
"""

import json
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from safetensors import safe_open


def read_all(directory: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                tensors[name] = f.get_tensor(name)
    return tensors


def group(name: str) -> str:
    return ".".join(name.split(".")[:2])  # model.language_model, model.visual, mtp.fc, ...


def compare(export: Path, base: Path) -> dict[str, object]:
    ours, theirs = read_all(export), read_all(base)
    differing: dict[str, int] = {}
    identical: dict[str, int] = {}
    dtype_mismatch = [n for n in theirs if n in ours and ours[n].dtype != theirs[n].dtype]
    max_abs = 0.0
    for name, base_tensor in theirs.items():
        if name not in ours:
            continue
        bucket = identical if torch.equal(ours[name], base_tensor) else differing
        bucket[group(name)] = bucket.get(group(name), 0) + 1
        if bucket is differing:
            diff = (ours[name].float() - base_tensor.float()).abs().max().item()
            max_abs = max(max_abs, diff)
    info = json.loads((export / "soe_export.json").read_text())
    return {
        "export": str(export),
        "base": str(base),
        "tensors_base": len(theirs),
        "missing_in_export": sorted(set(theirs) - set(ours)),
        "extra_in_export": sorted(set(ours) - set(theirs)),
        "dtype_mismatches": dtype_mismatch,
        "identical_by_group": dict(sorted(identical.items())),
        "differing_by_group": dict(sorted(differing.items())),
        "max_abs_difference": max_abs,
        "export_info": {
            k: info[k]
            for k in (
                "training",
                "merged_lora_modules",
                "trained_tensors",
                "trained_parameters",
                "trained_by_prefix",
                "tensors_changed_vs_base",
                "copied_from_base",
            )
        },
    }


class CheckExportCli(BaseSettings):
    model_config = SettingsConfigDict(cli_prog_name="scripts/check_export.py", cli_kebab_case=True)

    export: Path
    base: Path
    label: str
    report: Path

    def cli_cmd(self) -> None:
        result = compare(self.export, self.base)
        report = json.loads(self.report.read_text()) if self.report.exists() else {}
        report[self.label] = result
        self.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    CliApp.run(CheckExportCli)
