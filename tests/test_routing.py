from __future__ import annotations

from collections import Counter

import pytest

from loan_mlops.api.routing import assign_cohort


def test_no_challenger_always_returns_champion() -> None:
    for i in range(100):
        assert assign_cohort(f"app-{i}", challenger_pct=0.5, has_challenger=False) == "champion"


def test_zero_pct_always_returns_champion() -> None:
    for i in range(100):
        assert assign_cohort(f"app-{i}", challenger_pct=0.0, has_challenger=True) == "champion"


def test_full_pct_always_returns_challenger() -> None:
    for i in range(100):
        assert assign_cohort(f"app-{i}", challenger_pct=1.0, has_challenger=True) == "challenger"


def test_same_id_always_routes_to_same_cohort() -> None:
    """Determinism is the most important property — same applicant, same model."""
    for app_id in ["alice", "bob", "carol-99", "test-001"]:
        results = {
            assign_cohort(app_id, challenger_pct=0.5, has_challenger=True) for _ in range(20)
        }
        assert len(results) == 1, f"id {app_id} got mixed cohorts: {results}"


def test_traffic_split_approximates_target() -> None:
    """Over many IDs the actual split should be roughly the configured percentage."""
    counts: Counter = Counter()
    for i in range(2000):
        cohort = assign_cohort(f"app-{i}", challenger_pct=0.10, has_challenger=True)
        counts[cohort] += 1
    challenger_share = counts["challenger"] / 2000
    # 10% target, allow ±3pp tolerance with N=2000
    assert 0.07 < challenger_share < 0.13, f"got {challenger_share:.2%}"


def test_missing_id_is_handled() -> None:
    """No application_id is allowed — must not crash, must return a valid cohort."""
    for _ in range(10):
        result = assign_cohort(None, challenger_pct=0.5, has_challenger=True)
        assert result in {"champion", "challenger"}


@pytest.mark.parametrize("pct", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.99])
def test_split_works_at_various_percentages(pct: float) -> None:
    """Sanity: the function returns valid cohorts at any reasonable percentage."""
    cohorts = {
        assign_cohort(f"app-{i}", challenger_pct=pct, has_challenger=True) for i in range(200)
    }
    assert cohorts.issubset({"champion", "challenger"})
