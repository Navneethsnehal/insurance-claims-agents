"""Reviewer dashboard for a running claims API.

Run with ``streamlit run ui/dashboard.py``. The API base URL comes from
``CLAIMS_API_URL`` (default http://localhost:8000).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("CLAIMS_API_URL", "http://localhost:8000").rstrip("/")


def _get(path: str) -> httpx.Response:
    return httpx.get(f"{API_URL}{path}", timeout=30)


def _post(path: str, payload: dict[str, object]) -> httpx.Response:
    return httpx.post(f"{API_URL}{path}", json=payload, timeout=60)


def main() -> None:
    st.set_page_config(page_title="Claims review", layout="wide")
    st.title("Claims review")
    st.caption(f"API {API_URL}")

    try:
        health = _get("/health")
        health.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"API is not reachable: {exc}")
        return

    body = health.json()
    st.write(
        f"Provider `{body['llm_provider']}` · vector store `{body['vector_store']}` · "
        f"checkpointer `{body['checkpointer']}` · {body['clauses']} clauses"
    )

    samples = _get("/samples")
    samples.raise_for_status()
    catalog = samples.json()
    labels = {
        item["claim_id"]: f"{item['claim_id']} · {item['loss_type']} · {item['book_outcome']}"
        for item in catalog
    }

    with st.sidebar:
        st.header("Submit")
        choice = st.selectbox(
            "Sample file",
            options=list(labels),
            format_func=lambda key: labels[key],
        )
        selected = next(item for item in catalog if item["claim_id"] == choice)
        claim_id = st.text_input("Claim id", value=selected["claim_id"])
        if st.button("Run triage", type="primary"):
            response = _post(
                "/claims",
                {
                    "claim_id": claim_id,
                    "policy_id": selected["policy_id"],
                    "documents": [selected["document"]],
                },
            )
            if response.status_code >= 400:
                st.error(response.text)
            else:
                st.session_state["active"] = response.json()

    claims = _get("/claims")
    claims.raise_for_status()
    rows = claims.json()
    st.subheader("Files")
    if not rows:
        st.info("No claims yet. Submit a sample from the sidebar.")
    else:
        st.dataframe(rows, hide_index=True, use_container_width=True)
        active_id = st.selectbox("Open", options=[row["claim_id"] for row in rows])
        if st.button("Refresh file"):
            fetched = _get(f"/claims/{active_id}")
            fetched.raise_for_status()
            st.session_state["active"] = fetched.json()

    active = st.session_state.get("active")
    if not active:
        return

    st.subheader(f"{active['claim_id']} · {active['status']}")
    coverage = active.get("coverage") or {}
    fraud = active.get("fraud") or {}
    payout = active.get("payout") or {}
    memo = active.get("memo") or {}
    columns = st.columns(4)
    columns[0].metric("Coverage", coverage.get("outcome") or "—")
    columns[1].metric("Indemnity", f"{float(payout.get('indemnity') or 0):,.0f}")
    columns[2].metric("Fraud", f"{float(fraud.get('score') or 0):.2f}")
    columns[3].metric("Reserve", f"{float(payout.get('reserve') or 0):,.0f}")

    if active["status"] == "awaiting_human":
        reasons = ", ".join(active.get("escalation_reasons") or [])
        st.warning(f"Waiting for a reviewer. Reasons: {reasons}")
        with st.form("review"):
            decision = st.radio("Decision", options=["approve", "reject"], horizontal=True)
            reviewer = st.text_input("Reviewer", value="review.desk")
            notes = st.text_area("Notes")
            if st.form_submit_button("Submit decision"):
                response = _post(
                    f"/claims/{active['claim_id']}/resume",
                    {"decision": decision, "reviewer": reviewer, "notes": notes},
                )
                if response.status_code >= 400:
                    st.error(response.text)
                else:
                    st.session_state["active"] = response.json()
                    st.rerun()

    if memo:
        st.markdown("**Decision memo**")
        st.write(memo.get("narrative") or "")
        st.write(
            f"Recommendation `{memo.get('recommendation')}` · "
            f"final indemnity {float(memo.get('final_indemnity') or 0):,.2f}"
        )

    st.markdown("**Citations**")
    for citation in coverage.get("citations") or []:
        source = citation.get("source")
        st.markdown(f"`{citation['clause_id']}` · {citation['role']} · {source}")
        st.caption(citation.get("quote") or "")

    explanations = fraud.get("explanations") or []
    if explanations:
        st.markdown("**Fraud contributions**")
        st.bar_chart(
            {"delta": [float(row["delta"]) for row in explanations[:8]]},
            height=240,
        )
        st.dataframe(explanations[:8], hide_index=True, use_container_width=True)

    flags = fraud.get("flags") or []
    if flags:
        st.markdown("**Rule flags**")
        st.dataframe(flags, hide_index=True, use_container_width=True)

    st.markdown("**Trace**")
    st.dataframe(active.get("trace") or [], hide_index=True, use_container_width=True)
    totals = active.get("usage_totals") or {}
    st.caption(
        f"tokens in {totals.get('input_tokens', 0):.0f} · "
        f"out {totals.get('output_tokens', 0):.0f} · "
        f"cost ${totals.get('cost_usd', 0):.6f}"
    )


if __name__ == "__main__":
    main()
else:
    main()
