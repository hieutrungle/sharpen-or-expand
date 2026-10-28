"""Backport of vLLM PR #41602 ("Fix /wake_up crash on hybrid models", merged 2026-07-29) to the
vLLM 0.24.0 that verl 0.9.1 pins (decision 015).

With an FP8 KV cache, vLLM's wake-up from sleep mode calls `init_fp8_kv_scales`, which zeroes every
`kv_caches` entry as a tensor. Qwen3.5's linear-attention layers store a list of state tensors, so
the call fails ('list' object has no attribute 'zero_') before the first rollout (job 3222631).
The fix flattens the entries for the duration of the call. vLLM's own function still runs, and it
also resets the FP8 k/v scales to 1.0, the fixed scale evaluation uses (decision 009).

The patch is applied in every vLLM worker process by importing this module: verl's vLLM server
names a worker extension class, and soe's overrides point it at SoeColocateWorkerExtension.
GPU image only (imports vllm and verl).
"""

from typing import Any

import vllm
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from soe.workers.rollout.chat import flat_cache_tensors

PATCHED_VLLM = "0.24.0"  # drop this module when the pinned vLLM includes PR #41602
EXTENSION = "soe.workers.rollout.vllm_patches.SoeColocateWorkerExtension"

if vllm.__version__ != PATCHED_VLLM:
    raise RuntimeError(f"vllm_patches targets vLLM {PATCHED_VLLM}, found {vllm.__version__}")

_original_init_fp8_kv_scales = GPUModelRunner.init_fp8_kv_scales


def _init_fp8_kv_scales(self: Any) -> None:
    kv_caches = getattr(self, "kv_caches", [])
    self.kv_caches = flat_cache_tensors(kv_caches)
    try:
        _original_init_fp8_kv_scales(self)
    finally:
        self.kv_caches = kv_caches


if not getattr(GPUModelRunner.init_fp8_kv_scales, "_soe_patched", False):
    _init_fp8_kv_scales._soe_patched = True  # type: ignore[attr-defined]
    GPUModelRunner.init_fp8_kv_scales = _init_fp8_kv_scales


class SoeColocateWorkerExtension(vLLMColocateWorkerExtension):  # type: ignore[misc]
    """verl's colocate worker extension, unchanged; importing this module applies the patch."""
