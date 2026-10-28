"""The vLLM PR #41602 backport (decision 015): FP8 wake-up zeroes tensor and list KV-cache entries
and still resets the k/v scales. Container only (needs vllm, verl, torch); runs on a CPU node:
    sbatch slurm/cpu.sbatch python -m pytest -q \
        tests/workers/rollout/test_vllm_patches_in_container.py
The cases are those of the upstream PR's own test.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("verl")


def runner_with(kv_caches: list[object], cache_dtype: str = "fp8") -> object:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    import soe.workers.rollout.vllm_patches  # noqa: F401  (applies the patch)

    runner = Mock(spec=GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    runner.kv_caches = kv_caches
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.init_fp8_kv_scales = GPUModelRunner.init_fp8_kv_scales.__get__(runner, GPUModelRunner)
    return runner


def test_zeroes_tensor_and_list_entries_and_restores_the_list() -> None:
    full, states = torch.ones(4, 8), [torch.ones(2, 4), torch.ones(3, 6)]
    kv_caches = [None, full, states, None]
    runner = runner_with(kv_caches)
    runner.init_fp8_kv_scales()
    assert (full == 0).all() and all((t == 0).all() for t in states)
    assert runner.kv_caches is kv_caches  # the runner's own structure is put back


def test_noop_when_the_kv_cache_is_not_quantized() -> None:
    tensor = torch.ones(4, 8)
    runner = runner_with([tensor], cache_dtype="auto")
    runner.init_fp8_kv_scales()
    assert (tensor == 1).all()


def test_extension_class_is_verls_and_named_in_the_overrides() -> None:
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    from soe.workers.rollout.vllm_patches import EXTENSION, SoeColocateWorkerExtension

    assert issubclass(SoeColocateWorkerExtension, vLLMColocateWorkerExtension)
    module, name = EXTENSION.rsplit(".", 1)
    assert (module, name) == (SoeColocateWorkerExtension.__module__, "SoeColocateWorkerExtension")
