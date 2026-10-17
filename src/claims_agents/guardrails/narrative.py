"""Keep model prose from inventing clause identifiers."""

from __future__ import annotations

import re

_CLAUSE_TOKEN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")


def sanitize_narrative(text: str, allowed_clause_ids: set[str]) -> str:
    """Drop clause-shaped tokens that were not retrieved for this decision.

    Claim and policy identifiers (``CLM-``, ``POL-``) are left in place.
    """

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in allowed_clause_ids or token.startswith(("CLM-", "POL-")):
            return token
        return "[removed-citation]"

    return _CLAUSE_TOKEN.sub(replace, text).strip()
