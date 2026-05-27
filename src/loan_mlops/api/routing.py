"""Champion/challenger routing.

Routes traffic deterministically based on a hash of the application ID, so
the same applicant always sees the same model. This matters for two reasons:

1. Fairness — an applicant shouldn't see different decisions on retry
2. Debugging — when someone complains about a decline, we need to reproduce
   the exact prediction path
"""

from __future__ import annotations

import hashlib
from typing import Literal

Cohort = Literal["champion", "challenger"]


def assign_cohort(
    application_id: str | None,
    challenger_pct: float,
    has_challenger: bool,
) -> Cohort:
    """Decide which cohort an application falls into.

    If no challenger is configured, everyone gets champion. If application_id
    is missing (allowed by the schema), we fall back to a UUID so the same
    payload doesn't always hit champion — that would bias monitoring.
    """
    if not has_challenger or challenger_pct <= 0:
        return "champion"
    if challenger_pct >= 1.0:
        return "challenger"

    key = application_id or ""
    if not key:
        # Without an ID we can't be deterministic, but we shouldn't always
        # default to champion either — that biases comparisons.
        import uuid

        key = str(uuid.uuid4())

    # Hash → integer in [0, 1). Same key always produces same value.
    h = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:8], "big") / 2**64
    return "challenger" if bucket < challenger_pct else "champion"
