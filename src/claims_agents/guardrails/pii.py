"""Mask direct identifiers before documents are logged or sent to a model.

The named insured remains a claim attribute. Free-text SSNs, payment cards,
phone numbers, emails, and labeled dates of birth do not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?1[-.\s])?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\w)")
_DOB = re.compile(r"(?i)\b(?:dob|date of birth)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2})\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass(frozen=True)
class PiiFinding:
    kind: str
    count: int


def _luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[PiiFinding]]:
    """Replace identifier spans with stable tokens. Findings store counts only."""
    counts = {"email": 0, "ssn": 0, "phone": 0, "dob": 0, "card": 0}

    def email(match: re.Match[str]) -> str:
        counts["email"] += 1
        return "[REDACTED_EMAIL]"

    def ssn(match: re.Match[str]) -> str:
        counts["ssn"] += 1
        return "[REDACTED_SSN]"

    def phone(match: re.Match[str]) -> str:
        counts["phone"] += 1
        return "[REDACTED_PHONE]"

    def dob(match: re.Match[str]) -> str:
        counts["dob"] += 1
        prefix = match.group(0)[: match.start(1) - match.start(0)]
        return f"{prefix}[REDACTED_DOB]"

    def card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if not _luhn_ok(digits):
            return match.group(0)
        counts["card"] += 1
        return "[REDACTED_CARD]"

    redacted = _EMAIL.sub(email, text)
    redacted = _SSN.sub(ssn, redacted)
    redacted = _PHONE.sub(phone, redacted)
    redacted = _DOB.sub(dob, redacted)
    redacted = _CARD.sub(card, redacted)
    findings = [PiiFinding(kind, count) for kind, count in counts.items() if count]
    return redacted, findings
