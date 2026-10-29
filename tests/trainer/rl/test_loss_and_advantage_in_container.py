"""Hand-computed toy cases for verl 0.9.1's loss aggregation and GRPO advantages (M1.7).

These are the functions the soe configuration reaches: `core_algos.agg_loss` (called by
`compute_policy_loss_vanilla` from the FSDP engine's `ppo_loss`) and the v1 trainer's
`compute_advantage_for_multi_trajectories` (verl's GRPO estimator). Each check takes the
implementation as an argument, and each test also runs it on a deliberately broken variant, which
must fail: the toy values are chosen so that the usual silent bugs change them.

The prompt never enters the loss: verl sets `loss_mask = response_mask` over the generated tokens
only (trainer/ppo/v1/agent_loop_tq.py), and on the M1.5 and M1.6 runs verl's
`response_length/mean`, computed from that mask, equals the soe token records exactly at every step.

Image only (torch, verl).
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

AggLoss = Callable[..., torch.Tensor]

# Two sequences of unequal length (3 and 1 valid tokens) padded to 4; the padding holds 100s, so
# any weight on it shows.
LOSS = torch.tensor([[1.0, 2.0, 3.0, 100.0], [4.0, 100.0, 100.0, 100.0]])
MASK = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
HAND = {
    "token-mean": (1 + 2 + 3 + 4) / 4,  # 2.5
    "seq-mean-token-mean": ((1 + 2 + 3) / 3 + 4 / 1) / 2,  # 3.0
    "seq-mean-token-sum-norm": ((1 + 2 + 3) + 4) / 2 / 4,  # 1.25, normalised by 4
}


def check_aggregation(agg: AggLoss) -> None:
    for mode, value in HAND.items():
        extra = {"loss_scale_factor": 4} if mode == "seq-mean-token-sum-norm" else {}
        got = agg(loss_mat=LOSS, loss_mask=MASK, loss_agg_mode=mode, **extra)
        assert got.item() == pytest.approx(value, rel=1e-6), mode


def mask_off_by_one(**kwargs: torch.Tensor) -> torch.Tensor:
    kwargs["loss_mask"] = torch.roll(kwargs["loss_mask"], shifts=1, dims=-1)
    return agg_loss(**kwargs)


def padding_counted(**kwargs: torch.Tensor) -> torch.Tensor:
    kwargs["loss_mask"] = torch.ones_like(kwargs["loss_mask"])
    return agg_loss(**kwargs)


def test_aggregation_modes_match_hand_values() -> None:
    check_aggregation(agg_loss)
    for broken in (mask_off_by_one, padding_counted):
        with pytest.raises(AssertionError):
            check_aggregation(broken)


@pytest.mark.parametrize("mode", list(HAND))
def test_aggregation_is_invariant_to_ranks_and_micro_batches(mode: str) -> None:
    """The engine aggregates each micro-batch with the mini-batch's global counts, sums the
    micro-batches (gradient accumulation) and FSDP averages the ranks: the result must equal the
    mini-batch aggregated at once, whatever the split."""
    torch.manual_seed(0)
    lengths = [5, 1, 3, 7]
    width = max(lengths)
    loss = torch.randn(len(lengths), width)
    mask = torch.tensor([[1.0] * n + [0.0] * (width - n) for n in lengths])
    scale = {"loss_scale_factor": 16384} if mode == "seq-mean-token-sum-norm" else {}
    whole = agg_loss(loss_mat=loss, loss_mask=mask, loss_agg_mode=mode, **scale)
    ranks = [[[0], [1]], [[2, 3]]]  # rank 0: two micro-batches; rank 1: one
    counts = {"dp_size": 2, "batch_num_tokens": int(mask.sum()), "global_batch_size": 4}
    per_rank = []
    for micro_batches in ranks:
        total = torch.tensor(0.0)
        for rows in micro_batches:
            n = max(lengths[r] for r in rows)  # each micro-batch is padded to its own width
            total = total + agg_loss(
                loss_mat=loss[rows, :n],
                loss_mask=mask[rows, :n],
                loss_agg_mode=mode,
                **counts,
                **scale,
            )
        per_rank.append(total)
    assert torch.stack(per_rank).mean().item() == pytest.approx(whole.item(), rel=1e-5)


def test_dr_grpo_normalisation_needs_a_constant_scale_factor() -> None:
    """Without `loss_scale_factor`, seq-mean-token-sum-norm divides by the padded width of the
    micro-batch, which varies with dynamic batching: Dr.GRPO's configs must set it."""
    wider = torch.nn.functional.pad(LOSS, (0, 2)), torch.nn.functional.pad(MASK, (0, 2))
    mode = "seq-mean-token-sum-norm"
    narrow_default = agg_loss(loss_mat=LOSS, loss_mask=MASK, loss_agg_mode=mode)
    wide_default = agg_loss(loss_mat=wider[0], loss_mask=wider[1], loss_agg_mode=mode)
    assert narrow_default.item() != pytest.approx(wide_default.item())
    fixed = {"loss_agg_mode": mode, "loss_scale_factor": 16384}
    assert agg_loss(loss_mat=LOSS, loss_mask=MASK, **fixed).item() == pytest.approx(
        agg_loss(loss_mat=wider[0], loss_mask=wider[1], **fixed).item()
    )


# GRPO: group "a" has rewards 1, 0, 0, 1 (mean 0.5, unbiased std sqrt(1/3)); group "b" is all 1, a
# zero-variance group, which must give zero advantage, not NaN.
REWARDS = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
GROUPS = ["a"] * 4 + ["b"] * 4
LENGTHS = [3, 1, 2, 4, 4, 2, 1, 3]
EPS = 1e-6
STD_A = (1 / 3) ** 0.5
A = 0.5 / (STD_A + EPS)
GRPO_HAND = [A, -A, -A, A, 0.0, 0.0, 0.0, 0.0]
DR_GRPO_HAND = [0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0]

Advantage = Callable[[DataProto, list[str], bool], torch.Tensor]


def toy_batch() -> tuple[DataProto, list[str]]:
    width = max(LENGTHS)
    mask = torch.tensor([[1.0] * n + [0.0] * (width - n) for n in LENGTHS])
    rewards = torch.zeros_like(mask)
    for i, (n, r) in enumerate(zip(LENGTHS, REWARDS, strict=True)):
        rewards[i, n - 1] = r  # outcome reward on the last generated token
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards, "response_mask": mask},
        non_tensors={"uid": np.array(GROUPS, dtype=object)},
    )
    keys = [f"{g}_0_{i}" for i, g in enumerate(GROUPS)]  # {uid}_{session}_{output}
    return data, keys


def verl_advantages(data: DataProto, keys: list[str], norm_by_std: bool) -> torch.Tensor:
    out = compute_advantage_for_multi_trajectories(
        data, keys, AdvantageEstimator.GRPO, norm_adv_by_std_in_grpo=norm_by_std
    )
    advantages: torch.Tensor = out.batch["advantages"]
    return advantages


def check_advantages(advantage: Advantage) -> None:
    for norm_by_std, hand in ((True, GRPO_HAND), (False, DR_GRPO_HAND)):
        data, keys = toy_batch()
        mask = data.batch["response_mask"]
        adv = advantage(data, keys, norm_by_std)
        assert not torch.isnan(adv).any(), "NaN advantage"
        expected = torch.tensor(hand).unsqueeze(-1) * mask  # each valid token, none on padding
        assert torch.allclose(adv, expected, atol=1e-6), f"norm_by_std={norm_by_std}"


def std_without_eps(data: DataProto, keys: list[str], norm_by_std: bool) -> torch.Tensor:
    scores = data.batch["token_level_rewards"].sum(-1)
    groups = data.non_tensor_batch["uid"]
    out = torch.empty_like(scores)
    for g in set(groups):
        rows = torch.tensor([i for i, x in enumerate(groups) if x == g])
        s = scores[rows]
        out[rows] = (s - s.mean()) / s.std() if norm_by_std else s - s.mean()
    return out.unsqueeze(-1) * data.batch["response_mask"]


def dr_grpo_keeps_std(data: DataProto, keys: list[str], norm_by_std: bool) -> torch.Tensor:
    return verl_advantages(data, keys, True)


def test_grpo_and_dr_grpo_advantages_match_hand_values() -> None:
    check_advantages(verl_advantages)
    for broken in (std_without_eps, dr_grpo_keeps_std):
        with pytest.raises(AssertionError):
            check_advantages(broken)
