# src/agents/compliance_agent.py
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.screener_agent import RecruitmentState

# ── Mock registries (replace with real DB in production) ──────────────────────
DO_NOT_CONTACT = {"P0999", "P0888", "P0777"}

IRB_REGISTRY = {
    "NCT06710340": {
        "approved":  True,
        "protocol":  "IRB-2024-042",
        "expiry":    "2026-12-31",
        "sponsor":   "Raquel Ballarin"
    },
    "NCT05404061": {
        "approved":  True,
        "protocol":  "IRB-2023-118",
        "expiry":    "2026-06-30",
        "sponsor":   "Imperial College London"
    },
    "NCT05933174": {
        "approved":  True,
        "protocol":  "IRB-2024-007",
        "expiry":    "2026-09-15",
        "sponsor":   "University Hospitals"
    }
}


def compliance_agent(state: RecruitmentState) -> dict:
    """
    Safety veto gate — runs for ALL patients regardless of eligibility.
    Checks IRB approval, do-not-contact list, and message content.
    Can VETO outreach before it reaches any patient.
    """
    issues   = []
    approved = True

    # ── Hard check 1: Do-not-contact list ─────────────────────
    if state["patient_id"] in DO_NOT_CONTACT:
        return {
            "compliance_approved": False,
            "compliance_issues":   ["Patient is on do-not-contact list"],
            "audit_log": [{
                "step":    "compliance",
                "approved": False,
                "blocked_by": "do_not_contact_list",
                "patient_id": state["patient_id"]
            }]
        }

    # ── Hard check 2: IRB approval ────────────────────────────
    irb = IRB_REGISTRY.get(state["trial_id"])
    if not irb:
        issues.append(f"No IRB record found for trial {state['trial_id']}")
        approved = False
    elif not irb["approved"]:
        issues.append(f"Trial {state['trial_id']} does not have IRB approval")
        approved = False
    else:
        from datetime import datetime
        try:
            expiry = datetime.strptime(irb["expiry"], "%Y-%m-%d")
            if expiry < datetime.now():
                issues.append(f"IRB approval expired on {irb['expiry']}")
                approved = False
        except Exception:
            pass

    # If hard checks failed — stop here, no LLM needed
    if not approved:
        return {
            "compliance_approved": False,
            "compliance_issues":   issues,
            "audit_log": [{
                "step":     "compliance",
                "approved": False,
                "issues":   issues,
                "blocked_by": "irb_check"
            }]
        }

    # ── If no outreach message — approve (ineligible patients) ─
    if not state.get("outreach_message"):
        return {
            "compliance_approved": True,
            "compliance_issues":   [],
            "audit_log": [{
                "step":       "compliance",
                "approved":   True,
                "note":       "No outreach message — ineligible patient logged",
                "patient_id": state["patient_id"],
                "decision":   state["eligibility_decision"]
            }]
        }

    # ── Soft check: LLM content review ───────────────────────
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system = SystemMessage(content="""You are a strict clinical research compliance officer.

Review this patient outreach message against these rules:

MANDATORY CHECKS:
1. No coercive language ("you must", "you should", "don't miss")
2. No guaranteed outcomes ("will cure", "guaranteed to help", "proven to work")
3. Voluntary participation must be explicitly stated
4. Must not imply regular care will be affected if they don't join
5. No false urgency ("act now", "limited spots", "last chance")
6. No financial inducement language that could be coercive
7. Must be respectful and non-pressuring throughout

Respond with ONLY valid JSON:
{"approved": true, "issues": [], "risk_level": "low"}
or
{"approved": false, "issues": ["specific issue 1", "specific issue 2"], "risk_level": "high"}

Risk levels: low / medium / high""")

    user = HumanMessage(content=f"""
Please review this outreach message:

Trial: {state['trial_title']}
Patient summary: {state['patient_summary']}

Message:
{state['outreach_message']}
""")

    response = llm.invoke([system, user])

    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        review     = json.loads(json_match.group()) if json_match else {}
    except Exception:
        review = {"approved": True, "issues": [], "risk_level": "low"}

    content_approved = review.get("approved", True)
    content_issues   = review.get("issues",   [])
    risk_level       = review.get("risk_level", "low")

    if not content_approved:
        issues.extend(content_issues)
        approved = False

    return {
        "compliance_approved": approved,
        "compliance_issues":   issues,
        "audit_log": [{
            "step":           "compliance",
            "approved":       approved,
            "irb_protocol":   irb.get("protocol", "N/A") if irb else "N/A",
            "content_review": review,
            "risk_level":     risk_level,
            "issues":         issues,
            "patient_id":     state["patient_id"],
            "trial_id":       state["trial_id"]
        }]
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("\n" + "="*55)
    print("  COMPLIANCE AGENT — TEST RUNS")
    print("="*55)

    # ── Test 1: Normal eligible patient with message ──────────
    print("\n📋 Test 1: Normal eligible patient")
    print("-"*45)

    state_normal = {
        "patient_id":           "P0005",
        "patient_summary":      "45yo female, Type 2 diabetes, HbA1c 8.8%, on metformin",
        "trial_id":             "NCT06710340",
        "trial_title":          "Research on Effects of Nuts in Obese Diabetic Patients",
        "eligibility_decision": "ELIGIBLE",
        "outreach_message":     """Dear Patient,
We are reaching out about a research study on the effects of nuts on 
blood sugar in people with Type 2 diabetes. Participation is completely 
voluntary and will not affect your regular care. If interested, please 
contact our research team to learn more. Thank you for your consideration.""",
        "criteria":             [],
        "screening_results":    [],
        "eligibility_score":    1.0,
        "compliance_approved":  False,
        "compliance_issues":    [],
        "audit_log":            []
    }

    result1 = compliance_agent(state_normal)
    status  = "✅ APPROVED" if result1["compliance_approved"] else "❌ REJECTED"
    print(f"  Result   : {status}")
    print(f"  Issues   : {result1['compliance_issues']}")
    print(f"  Log      : {result1['audit_log'][0]}")

    # ── Test 2: Patient on do-not-contact list ─────────────────
    print("\n📋 Test 2: Patient on do-not-contact list")
    print("-"*45)

    state_dnc = {**state_normal, "patient_id": "P0999"}
    result2   = compliance_agent(state_dnc)
    status    = "✅ APPROVED" if result2["compliance_approved"] else "❌ REJECTED"
    print(f"  Result   : {status}")
    print(f"  Issues   : {result2['compliance_issues']}")

    # ── Test 3: Coercive message that should be rejected ───────
    print("\n📋 Test 3: Coercive message — should be REJECTED")
    print("-"*45)

    state_coercive = {
        **state_normal,
        "patient_id": "P0010",
        "outreach_message": """You MUST join this trial immediately. 
This is your only chance to get better. Limited spots available — 
act now or miss out forever. You should definitely sign up today. 
This trial WILL cure your diabetes."""
    }

    result3 = compliance_agent(state_coercive)
    status  = "✅ APPROVED" if result3["compliance_approved"] else "❌ REJECTED"
    print(f"  Result   : {status}")
    print(f"  Issues   : {result3['compliance_issues']}")
    print(f"  Risk     : {result3['audit_log'][0].get('risk_level', 'N/A')}")

    # ── Test 4: Ineligible patient — no message ────────────────
    print("\n📋 Test 4: Ineligible patient — no message")
    print("-"*45)

    state_inelig = {
        **state_normal,
        "patient_id":           "P0020",
        "eligibility_decision": "INELIGIBLE",
        "outreach_message":     ""
    }

    result4 = compliance_agent(state_inelig)
    status  = "✅ APPROVED" if result4["compliance_approved"] else "❌ REJECTED"
    print(f"  Result   : {status}")
    print(f"  Note     : {result4['audit_log'][0].get('note', '')}")

    print("\n" + "="*55)
    print("  ALL COMPLIANCE TESTS COMPLETE")
    print("="*55)