"""`python -m claims_agents.evals` writes reports/ and exits non-zero on a gate miss."""

from __future__ import annotations

import json
from pathlib import Path

from claims_agents.evals.harness import run_evaluation, write_report


def main() -> None:
    report = run_evaluation()
    destination = Path("reports")
    write_report(report, destination)
    summary = {
        "passed": report["passed"],
        "failures": report["failures"],
        "metrics": report["metrics"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
