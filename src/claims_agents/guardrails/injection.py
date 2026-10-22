"""Detect prompt-injection attempts in uploaded loss documents.

This is a control, not a claim of complete coverage. High-severity hits force
human review and the matched span is removed from the model prompt. Coverage
is still decided from parsed facts, not from instructions in the document.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "ignore_instructions",
        re.compile(r"ignore\s+(?:all|any|previous|prior)\s+instructions", re.I),
        "high",
    ),
    (
        "disregard_policy",
        re.compile(r"disregard\s+(?:the\s+)?(?:policy|exclusions|guidelines|coverage)", re.I),
        "high",
    ),
    ("role_override", re.compile(r"you\s+are\s+now", re.I), "high"),
    ("system_tag", re.compile(r"<\s*/?\s*(?:system|assistant)\s*>", re.I), "high"),
    ("jailbreak", re.compile(r"\b(?:do anything now|dan mode)\b", re.I), "high"),
    (
        "exfiltrate",
        re.compile(r"\b(?:exfiltrate|reveal the (?:system|hidden) prompt)\b", re.I),
        "high",
    ),
    (
        "override_decision",
        re.compile(r"override\s+(?:the\s+)?(?:coverage|decision|exclusion)", re.I),
        "high",
    ),
)


@dataclass(frozen=True)
class InjectionFinding:
    code: str
    severity: str


def detect_injections(text: str) -> list[InjectionFinding]:
    findings: list[InjectionFinding] = []
    for code, pattern, severity in _PATTERNS:
        if pattern.search(text):
            findings.append(InjectionFinding(code=code, severity=severity))
    return findings


def strip_injections(text: str) -> tuple[str, list[InjectionFinding]]:
    findings = detect_injections(text)
    cleaned = text
    for _code, pattern, _severity in _PATTERNS:
        cleaned = pattern.sub("[INJECTION_REMOVED]", cleaned)
    return cleaned, findings


def has_high_severity(findings: Sequence[object]) -> bool:
    for finding in findings:
        severity: object
        if isinstance(finding, InjectionFinding):
            severity = finding.severity
        elif isinstance(finding, dict):
            severity = finding.get("severity")
        else:
            continue
        if severity == "high":
            return True
    return False
