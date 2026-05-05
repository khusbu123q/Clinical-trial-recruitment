# src/agents/orchestrator.py
import json
import os
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.agents.screener_agent import (
    RecruitmentState,
    screener_agent,
    load_patient_db,
    PATIENT_DB
)
from src.agents.communicator_agent import communicator_agent
from src.agents.compliance_agent import compliance_agent

load_dotenv()


# ── Routing logic ─────────────────────────────────────────────────────────────
def route_after_screening(
    state: RecruitmentState
) -> Literal["communicator", "compliance"]:
    """
    After screening:
    - ELIGIBLE   → communicator (generate outreach) → compliance
    - INELIGIBLE → compliance directly (just log)
    - UNCERTAIN  → compliance directly (flag for human review)
    """
    if state["eligibility_decision"] == "ELIGIBLE":
        return "communicator"
    return "compliance"


def route_after_compliance(
    state: RecruitmentState
) -> Literal["end"]:
    """Always end after compliance — no more steps."""
    return "end"


# ── Build the LangGraph ───────────────────────────────────────────────────────
def build_recruitment_graph() -> StateGraph:
    """
    Constructs the multi-agent LangGraph workflow.

    Flow:
    START → screener → [ELIGIBLE] → communicator → compliance → END
                     → [INELIGIBLE/UNCERTAIN] → compliance → END
    """
    graph = StateGraph(RecruitmentState)

    # Add agent nodes
    graph.add_node("screener",     screener_agent)
    graph.add_node("communicator", communicator_agent)
    graph.add_node("compliance",   compliance_agent)

    # Set entry point
    graph.set_entry_point("screener")

    # Conditional routing after screener
    graph.add_conditional_edges(
        "screener",
        route_after_screening,
        {
            "communicator": "communicator",
            "compliance":   "compliance"
        }
    )

    # Communicator always goes to compliance
    graph.add_edge("communicator", "compliance")

    # Compliance always ends
    graph.add_edge("compliance", END)

    return graph.compile()


# ── Run one patient through the full pipeline ─────────────────────────────────
def run_patient(
    app,
    patient_id:  str,
    trial:       dict,
    criteria:    list[dict]
) -> RecruitmentState:
    """Run one patient through the complete multi-agent pipeline."""

    patient = PATIENT_DB.get(patient_id, {})

    # Build patient summary from features
    age  = patient.get("age", "?")
    sex  = patient.get("sex", "?")
    dx   = patient.get("diagnoses", [])
    meds = patient.get("medications", [])
    labs = patient.get("lab_values", {})

    summary = (
        f"{age}yo {sex}, "
        f"diagnoses: {', '.join(dx) if dx else 'none'}, "
        f"medications: {', '.join(meds) if meds else 'none'}, "
        f"HbA1c: {labs.get('hba1c', 'N/A')}%, "
        f"eGFR: {labs.get('egfr', 'N/A')}, "
        f"BMI: {labs.get('bmi', 'N/A')}"
    )

    initial_state: RecruitmentState = {
        "patient_id":           patient_id,
        "patient_summary":      summary,
        "patient_features":     patient,
        "trial_id":             trial.get("nct_id", ""),
        "trial_title":          trial.get("title", ""),
        "criteria":             criteria,
        "screening_results":    [],
        "eligibility_decision": "",
        "eligibility_score":    0.0,
        "outreach_message":     "",
        "compliance_approved":  False,
        "compliance_issues":    [],
        "audit_log":            []
    }

    result = app.invoke(initial_state)
    return result


def print_result(result: dict):
    """Print a formatted result for one patient."""
    decision   = result.get("eligibility_decision", "")
    score      = result.get("eligibility_score", 0)
    approved   = result.get("compliance_approved", False)
    message    = result.get("outreach_message", "")
    audit_log  = result.get("audit_log", [])

    icons = {
        "ELIGIBLE":   "✅",
        "INELIGIBLE": "❌",
        "UNCERTAIN":  "⚠️"
    }
    icon = icons.get(decision, "❓")

    print(f"\n  {icon} {result['patient_id']} → {decision} (score: {score:.2f})")

    # Show screening results
    for r in result.get("screening_results", [])[:3]:
        met  = "✓" if r.get("met") else "✗"
        text = r.get("criterion", r.get("reasoning", ""))[:60]
        print(f"     {met} {text}")

    # Show compliance
    comp_icon = "✅" if approved else "❌"
    print(f"     Compliance: {comp_icon}", end="")
    if result.get("compliance_issues"):
        print(f" — {result['compliance_issues'][0]}")
    else:
        print()

    # Show message preview if generated
    if message:
        words = message.split()
        preview = " ".join(words[:20])
        print(f"     Message: \"{preview}...\" ({len(words)} words)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load patient database
    load_patient_db()

    # Load trial and criteria
    trials_path = "/Users/khusbuagarwal/Desktop/clinical_trial_recruitment/data/processed/type_2_diabetes_parsed.json"

    with open(trials_path) as f:
        trials = json.load(f)

    # Use trial 2 — NCT06710340 (nuts study — had eligible patients)
    trial    = trials[1]
    criteria = trial.get("inclusion", []) + trial.get("exclusion", [])

    print(f"\n{'='*58}")
    print(f"  FULL MULTI-AGENT PIPELINE — ORCHESTRATOR TEST")
    print(f"{'='*58}")
    print(f"  Trial    : {trial['nct_id']}")
    print(f"  Title    : {trial['title'][:50]}...")
    print(f"  Criteria : {len(criteria)}")
    print(f"{'='*58}")

    # Build the graph
    app = build_recruitment_graph()
    print(f"\n✅ LangGraph compiled successfully")
    print(f"   Flow: screener → [eligible] → communicator → compliance")
    print(f"         screener → [ineligible] → compliance")

    # Run 5 patients through the complete pipeline
    test_patients = ["P0001", "P0005", "P0010", "P0015", "P0020"]

    print(f"\n{'='*58}")
    print(f"  RUNNING {len(test_patients)} PATIENTS THROUGH FULL PIPELINE")
    print(f"{'='*58}")

    results     = []
    eligible    = 0
    ineligible  = 0
    uncertain   = 0
    approved    = 0

    for i, pid in enumerate(test_patients):
        print(f"\n  [{i+1}/{len(test_patients)}] Processing {pid}...")
        try:
            result = run_patient(app, pid, trial, criteria)
            results.append(result)

            decision = result.get("eligibility_decision", "")
            if decision == "ELIGIBLE":   eligible  += 1
            elif decision == "INELIGIBLE": ineligible += 1
            else:                          uncertain  += 1

            if result.get("compliance_approved"):
                approved += 1

            print_result(result)

        except Exception as e:
            print(f"  ❌ Error processing {pid}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*58}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*58}")
    print(f"  Patients processed : {len(results)}")
    print(f"  Eligible           : {eligible}")
    print(f"  Ineligible         : {ineligible}")
    print(f"  Uncertain          : {uncertain}")
    print(f"  Compliance approved: {approved}")
    print(f"{'='*58}")

    # Save full results
    out_path = "/Users/khusbuagarwal/Desktop/clinical_trial_recruitment/data/processed/agent_results.json"

    with open(out_path, "w") as f:
        json.dump(
            [{
                "patient_id":           r.get("patient_id"),
                "trial_id":             r.get("trial_id"),
                "eligibility_decision": r.get("eligibility_decision"),
                "eligibility_score":    r.get("eligibility_score"),
                "compliance_approved":  r.get("compliance_approved"),
                "compliance_issues":    r.get("compliance_issues"),
                "outreach_message":     r.get("outreach_message", ""),
                "screening_results":    r.get("screening_results", []),
                "audit_log":            r.get("audit_log", [])
            } for r in results],
            f, indent=2
        )

    print(f"\n💾 Saved full results → {out_path}")
    print(f"✅ Orchestrator complete")