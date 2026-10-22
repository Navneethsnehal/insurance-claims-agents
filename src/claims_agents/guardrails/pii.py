"""Mask direct identifiers before documents are logged or sent to a model.

The named insured remains a claim attribute. Free-text SSNs, payment cards,
phone numbers, emails, and labeled dates of birth do not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# A parenthesized area code may sit against the exchange: "(214)555-0198".
# The separator after ")" is optional. A bare area code still needs one,
# so a 10-digit claim amount is not treated as a phone number.
_PHONE = re.compile(
    r"(?<!\w)"
    r"(?:\+?1[-.\s]?)?"
    r"(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])"
    r"\d{3}[-.\s]\d{4}"
    r"(?!\w)"
)
_DOB = re.compile(r"(?i)\b(?:dob|date of birth)\s*[:#-]?\s*(\d{4}-\d{2}-\d{2})\b")
# Longer than 19 digits so a valid card followed by more digits is not one
# failed Luhn check. Spans are split below.
_CARD_RUN = re.compile(r"(?<!\d)\d(?:[ -]*\d){12,}(?!\d)")


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

    redacted = _EMAIL.sub(email, text)
    redacted = _SSN.sub(ssn, redacted)
    redacted = _PHONE.sub(phone, redacted)
    redacted = _DOB.sub(dob, redacted)
    redacted, card_count = _redact_cards(redacted)
    counts["card"] += card_count
    findings = [PiiFinding(kind, count) for kind, count in counts.items() if count]
    return redacted, findings


def _redact_cards(text: str) -> tuple[str, int]:
    """Redact Luhn-valid spans of 13 to 19 digits.

    A separator-joined run is checked as a whole and, when that fails, as a
    shorter prefix. ``4111111111111111 123`` must not survive just because the
    19-digit concatenation fails Luhn. A single digit string is checked only
    at its full length, so an arbitrary window inside a non-card reference
    number is left alone.
    """
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        updated, found = _redact_luhn_spans(match.group(0))
        count += found
        return updated

    return _CARD_RUN.sub(replace, text), count


def _redact_luhn_spans(raw: str) -> tuple[str, int]:
    tokens = re.findall(r"\d+|[^\d]+", raw)
    count = 0
    pieces: list[str] = []
    index = 0
    while index < len(tokens):
        if not tokens[index].isdigit():
            pieces.append(tokens[index])
            index += 1
            continue
        end = _luhn_span_end(tokens, index)
        if end is None:
            pieces.append(tokens[index])
            index += 1
            continue
        pieces.append("[REDACTED_CARD]")
        count += 1
        index = end + 1
    return "".join(pieces), count


def _luhn_span_end(tokens: list[str], start: int) -> int | None:
    digits = ""
    last_good: int | None = None
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token.isdigit():
            digits += token
        elif re.fullmatch(r"[ -]+", token) is None:
            break
        if len(digits) > 19:
            break
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            last_good = index
    return last_good
