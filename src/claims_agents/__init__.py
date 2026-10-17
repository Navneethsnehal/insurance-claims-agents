"""Insurance claims triage agents.

A supervisor-routed LangGraph workflow that extracts a first notice of loss,
decides coverage against the policy form, scores fraud, and estimates the
reserve. High-risk files stop for a human decision before the memo is issued.
"""

__version__ = "0.1.0"
