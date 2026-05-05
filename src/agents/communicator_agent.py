# src/agents/communicator_agent.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.screener_agent import RecruitmentState


def communicator_agent(state: RecruitmentState) -> dict:
    """
    Generates a personalised patient outreach message.
    Only runs if patient is ELIGIBLE.
    Uses GPT-4o with strict constraints.
    """
    if state["eligibility_decision"] != "ELIGIBLE":
        return {
            "outreach_message": "",
            "audit_log": [{
                "step":    "communicator",
                "skipped": True,
                "reason":  f"Patient {state['eligibility_decision']} — no message needed"
            }]
        }

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Build supporting reasons string
    supporting = [
        r.get("reasoning", "")
        for r in state["screening_results"]
        if r.get("met") and "NOT MET" not in r.get("reasoning", "")
    ]
    reasons_text = "\n".join(f"- {r}" for r in supporting[:5])

    system = SystemMessage(content="""You are a compassionate clinical research coordinator
writing a patient outreach message about a clinical trial.

STRICT RULES:
1. Maximum 180 words
2. Plain language — 8th grade reading level
3. Never guarantee outcomes or promise benefits
4. Always state participation is completely voluntary
5. Never use pressure language like "you must" or "you should"
6. Always say it will not affect their regular medical care
7. Include a clear next step (contact research team)
8. Warm and respectful tone throughout

Structure:
- Opening: why we are reaching out
- Middle: what the trial involves briefly
- End: next steps and voluntary nature""")

    user = HumanMessage(content=f"""
Write a patient outreach message for:

Patient summary: {state['patient_summary']}
Trial: {state['trial_title']}
Trial ID: {state['trial_id']}

Why this patient may qualify:
{reasons_text}

Write the complete outreach message now.
""")

    response = llm.invoke([system, user])
    message  = response.content.strip()

    # Self-critique step
    critique_system = SystemMessage(content="""You are a clinical research compliance reviewer.
Review this patient outreach message and check for:
1. Any coercive or pressure language
2. Any false promises or guaranteed outcomes
3. Any unclear voluntary participation language
4. Reading level too complex

Respond with ONLY JSON:
{"approved": true/false, "issues": ["issue1", "issue2"]}

If no issues found: {"approved": true, "issues": []}""")

    critique_user = HumanMessage(content=message)
    critique_response = llm.invoke([critique_system, critique_user])

    import json
    import re
    try:
        json_match = re.search(r'\{.*\}', critique_response.content, re.DOTALL)
        critique   = json.loads(json_match.group()) if json_match else {"approved": True, "issues": []}
    except Exception:
        critique = {"approved": True, "issues": []}

    # If critique found issues — regenerate with explicit fixes
    if not critique.get("approved") and critique.get("issues"):
        issues_text = "\n".join(f"- {i}" for i in critique["issues"])
        fix_user = HumanMessage(content=f"""
Rewrite the message fixing these issues:
{issues_text}

Original message:
{message}

Write the corrected message only.
""")
        fixed_response = llm.invoke([system, fix_user])
        message        = fixed_response.content.strip()

    return {
        "outreach_message": message,
        "audit_log": [{
            "step":             "communicator",
            "message_length":   len(message.split()),
            "self_critique":    critique,
            "message_generated": True
        }]
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.agents.screener_agent import load_patient_db, PATIENT_DB

    load_patient_db()

    # Test with an ELIGIBLE patient state
    test_state = {
        "patient_id":           "P0005",
        "patient_summary":      "45yo female, Type 2 diabetes, HbA1c 8.8%, on metformin, BMI 33.2, no insulin",
        "patient_features":     PATIENT_DB.get("P0005", {}),
        "trial_id":             "NCT06710340",
        "trial_title":          "Research on Effects of Acute Consumption of Nuts in Obese Diabetic Patients",
        "criteria":             [],
        "screening_results": [
            {
                "criterion": "Diagnosis of non-insulin-dependent type II diabetes",
                "met":       True,
                "reasoning": "Patient has Type 2 diabetes — criterion MET"
            },
            {
                "criterion": "Women over 18 years of age",
                "met":       True,
                "reasoning": "Patient is female, age 45 — criterion MET"
            }
        ],
        "eligibility_decision": "ELIGIBLE",
        "eligibility_score":    1.0,
        "outreach_message":     "",
        "compliance_approved":  False,
        "compliance_issues":    [],
        "audit_log":            []
    }

    print("\n" + "="*55)
    print("  COMMUNICATOR AGENT — TEST RUN")
    print("="*55)
    print(f"  Patient  : {test_state['patient_id']}")
    print(f"  Summary  : {test_state['patient_summary']}")
    print(f"  Decision : {test_state['eligibility_decision']}")
    print("="*55)

    result = communicator_agent(test_state)

    print(f"\n  GENERATED MESSAGE:")
    print(f"  {'-'*51}")
    print(result["outreach_message"])
    print(f"  {'-'*51}")
    print(f"\n  Word count : {len(result['outreach_message'].split())}")
    print(f"\n  AUDIT LOG:")
    for log in result["audit_log"]:
        print(f"  {log}")
    print("="*55)