import pytest

from soe.utils.dataset.problems import Level
from soe.utils.dataset.records import SampleRecord
from soe.utils.metric.eval_tables import problem_result, tier_table
from soe.utils.metric.pass_at_k import pass_at_k, same_string


def records(pid: str, correct: list[bool], truncated: int = 0) -> list[SampleRecord]:
    return [
        SampleRecord(
            problem_id=pid,
            sample_index=i,
            checkpoint="m",
            completion="",
            extracted_answer=None if i < truncated else ("1" if ok else "2"),
            correct=ok,
            completion_tokens=10,
            finish_reason="length" if i < truncated else "stop",
        )
        for i, ok in enumerate(correct)
    ]


def result(pid: str, correct: list[bool], hard: bool = False, truncated: int = 0):  # type: ignore[no-untyped-def]
    return problem_result(
        records(pid, correct, truncated)[::-1],  # order must not matter
        problem_set="s",
        level=Level.A,
        hard=hard,
        n_main=4,
        maj_ks=[1, 4, 8],
        same=same_string,
    )


def test_main_counts_use_the_first_samples_and_all_counts_use_every_sample() -> None:
    r = result("p", [False] * 4 + [True] * 4, hard=True)
    assert (r.main.n, r.main.correct, r.all.n, r.all.correct) == (4, 0, 8, 4)
    assert r.maj == {1: False, 4: False, 8: False}  # tie at 8: the first group ("2") wins


def test_truncation_and_boxed_counts() -> None:
    r = result("p", [False, False, True, True], truncated=2)
    assert (r.main.truncated, r.main.boxed, r.main.stopped, r.main.boxed_when_stopped) == (
        2,
        2,
        2,
        2,
    )
    assert 8 not in r.maj  # k above the sample count is skipped


def test_tier_tables_average_over_problems() -> None:
    easy = result("a", [True, True, False, False])
    hard = result("b", [False] * 4 + [True] * 4, hard=True)
    main = tier_table([easy, hard], [1, 4], hard=False)
    assert main["problems"] == 2
    assert main["pass@1"] == pytest.approx(round((0.5 + 0.0) / 2, 4))
    assert main["pass@4"] == pytest.approx(0.5)
    # "a" answers 1,1,2,2: the tie goes to the first group, "1", which is correct; "b" is wrong
    assert main["maj@4"] == pytest.approx(0.5)
    top = tier_table([hard], [1, 8], hard=True)
    assert top["pass@8"] == 1.0 and top["pass@1"] == pytest.approx(round(pass_at_k(8, 4, 1), 4))


def test_rejects_gaps_in_sample_indices() -> None:
    gappy = records("p", [True] * 5)
    del gappy[2]
    with pytest.raises(ValueError, match="sample indices"):
        problem_result(
            gappy,
            problem_set="s",
            level=Level.A,
            hard=False,
            n_main=4,
            maj_ks=[1],
            same=same_string,
        )
