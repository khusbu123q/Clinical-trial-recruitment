# src/agents/screener_agent.py
import json
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage


# ── Shared state ──────────────────────────────────────────────────────────────
class RecruitmentState(TypedDict):
    patient_id:           str
    patient_summary:      str
    patient_features:     dict
    trial_id:             str
    trial_title:          str
    criteria:             list[dict]
    screening_results:    Annotated[list, operator.add]
    eligibility_decision: str        # ELIGIBLE / INELIGIBLE / UNCERTAIN
    eligibility_score:    float
    outreach_message:     str
    compliance_approved:  bool
    compliance_issues:    list[str]
    audit_log:            Annotated[list, operator.add]


# ── Patient data store (mock — replace with DB in production) ─────────────────
PATIENT_DB = {}


def load_patient_db():
    """Load patient features into memory for tool access."""
    import os
    import json
    path = "/Users/khusbuagarwal/Desktop/clinical_trial_recruitment/data/processed/patient_features.json"
    if os.path.exists(path):
        with open(path) as f:
            patients = json.load(f)
        for p in patients:
            PATIENT_DB[p["patient_id"]] = p
        print(f"✅ Loaded {len(PATIENT_DB)} patients into DB")
    else:
        print("⚠️  patient_features.json not found — using mock data")
        # Mock patient for testing
        PATIENT_DB["P0001"] = {
            "patient_id":  "P0001",
            "age":         58,
            "sex":         "M",
            "diagnoses":   ["type_2_diabetes", "hypertension"],
            "medications": ["metformin", "lisinopril"],
            "lab_values":  {"hba1c": 8.2, "egfr": 72.0,
                            "creatinine": 1.1, "bmi": 31.4},
            "vitals":      {"systolic_bp": 138.0, "diastolic_bp": 86.0}
        }


# ── Tools the screener agent can call ────────────────────────────────────────
@tool
def get_patient_labs(patient_id: str, lab_name: str) -> str:
    """
    Get a specific lab value for a patient.
    lab_name options: hba1c, egfr, creatinine, bmi,
                      cholesterol, glucose, hemoglobin
    Returns the value and unit as a string.
    """
    patient = PATIENT_DB.get(patient_id)
    if not patient:
        return f"Patient {patient_id} not found"

    labs = patient.get("lab_values", {})
    val  = labs.get(lab_name.lower())

    if val is None:
        return f"{lab_name} not recorded for {patient_id}"

    units = {
        "hba1c": "%", "egfr": "mL/min",
        "creatinine": "mg/dL", "bmi": "kg/m2",
        "cholesterol": "mg/dL", "glucose": "mg/dL",
        "hemoglobin": "g/dL"
    }
    unit = units.get(lab_name.lower(), "")
    return f"{lab_name}: {val} {unit}".strip()


@tool
def get_patient_medications(patient_id: str) -> str:
    """
    Get the list of current medications for a patient.
    Returns a comma-separated list of medication names.
    """
    patient = PATIENT_DB.get(patient_id)
    if not patient:
        return f"Patient {patient_id} not found"

    meds = patient.get("medications", [])
    if not meds:
        return f"{patient_id} has no recorded medications"

    return f"Current medications: {', '.join(meds)}"

@tool
def get_patient_diagnoses(patient_id: str) -> str:
    """
    Get the list of active diagnoses for a patient.
    Returns a comma-separated list of diagnosis names.
    """
    patient = PATIENT_DB.get(patient_id)
    if not patient:
        return f"Patient {patient_id} not found"

    dx = patient.get("diagnoses", [])
    if not dx:
        return f"{patient_id} has no recorded diagnoses"

    # Return both underscore and space versions for easy matching
    readable = [d.replace("_", " ") for d in dx]
    return f"Active diagnoses: {', '.join(readable)}"
@tool
def get_patient_demographics(patient_id: str) -> str:
    """
    Get age, sex, and vitals for a patient.
    """
    patient = PATIENT_DB.get(patient_id)
    if not patient:
        return f"Patient {patient_id} not found"

    age    = patient.get("age", "unknown")
    sex    = patient.get("sex", "unknown")
    vitals = patient.get("vitals", {})
    sbp    = vitals.get("systolic_bp", "not recorded")
    dbp    = vitals.get("diastolic_bp", "not recorded")
    bmi    = patient.get("lab_values", {}).get("bmi", "not recorded")

    return (
        f"Age: {age} years | Sex: {sex} | "
        f"BP: {sbp}/{dbp} mmHg | BMI: {bmi} kg/m2"
    )


# ── Screener agent node ───────────────────────────────────────────────────────
def screener_agent(state: RecruitmentState) -> dict:
    """
    Evaluates each trial criterion against patient data.
    Uses ReAct reasoning — thinks, calls tools, observes, concludes.
    """
    tools = [
        get_patient_labs,
        get_patient_medications,
        get_patient_diagnoses,
        get_patient_demographics,
    ]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    system = SystemMessage(content="""You are a clinical trial eligibility screener.

Your job is to evaluate whether a patient meets each trial criterion.

For each criterion:
1. Think about what patient data you need
2. Call the appropriate tool to get that data
3. Reason whether the criterion is met or not
4. Be precise — cite the exact value you found

After evaluating ALL criteria, respond with ONLY a JSON array like this:
[
  {
    "criterion": "criterion text here",
    "met": true,
    "reasoning": "HbA1c is 8.2% which is >= 7.5% — criterion MET"
  }
]

Rules:
- For INCLUSION criteria: met=true means patient qualifies
- For EXCLUSION criteria: met=true means patient is BLOCKED
- If you cannot find the data: met=false with reasoning explaining data is missing
- For INCLUSION criteria that specify a specific sex (male/female): evaluate carefully against patient demographics
- Always use tools before concluding
""")

    criteria_text = "\n".join([
        f"- [{c.get('criterion_type','').upper()}] {c.get('raw_text', c.get('value',''))}"
        for c in state["criteria"]
    ])

    user = HumanMessage(content=f"""
Patient ID: {state['patient_id']}
Patient summary: {state['patient_summary']}

Please evaluate ALL of these criteria:
{criteria_text}

Use tools to look up the patient's actual data before concluding.
""")

    # ReAct loop — runs until model stops calling tools
    messages = [system, user]
    max_iterations = 8
    iteration = 0

    while iteration < max_iterations:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        iteration += 1

        # Check if model wants to call tools
        if not response.tool_calls:
            break

        # Execute each tool call
        from langchain_core.messages import ToolMessage
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_map  = {t.name: t for t in tools}

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Tool {tool_name} not found"

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"]
            ))

    # Parse the final response
    final_text = response.content if hasattr(response, "content") else ""

    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', final_text, re.DOTALL)
        if json_match:
            results = json.loads(json_match.group())
        else:
            results = []
    except Exception:
        results = []

    # If parsing failed — create basic results
    if not results:
        results = [{
            "criterion": c.get("raw_text", ""),
            "met":       True,
            "reasoning": "Could not parse agent response — flagged for review"
        } for c in state["criteria"]]

    # Determine eligibility
    inclusion_criteria = [
        c for c in state["criteria"]
        if c.get("criterion_type") == "inclusion"
    ]
    exclusion_criteria = [
        c for c in state["criteria"]
        if c.get("criterion_type") == "exclusion"
    ]

    inc_results = results[:len(inclusion_criteria)]
    exc_results = results[len(inclusion_criteria):]

    all_inclusion_met = all(r.get("met", False) for r in inc_results)
    any_exclusion_met = any(r.get("met", False) for r in exc_results)

    if all_inclusion_met and not any_exclusion_met:
        decision = "ELIGIBLE"
    elif any_exclusion_met:
        decision = "INELIGIBLE"
    else:
        decision = "UNCERTAIN"

    score = (
        sum(1 for r in inc_results if r.get("met", False)) /
        max(len(inc_results), 1)
    )

    return {
        "screening_results": results,
        "eligibility_decision": decision,
        "eligibility_score": round(score, 3),
        "audit_log": [{
            "step":     "screener",
            "decision": decision,
            "score":    score,
            "criteria_evaluated": len(results)
        }]
    }


# ── Test the screener agent standalone ───────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    load_patient_db()

    test_state = {
        "patient_id": "P0001",
        "patient_summary": "58yo male, Type 2 diabetes, HbA1c 8.2%, on metformin, eGFR 72",
        "patient_features": PATIENT_DB.get("P0001", {}),
        "trial_id": "NCT06710340",
        "trial_title": "Research on Effects of Nuts in Obese Diabetic Patients",
        "criteria": [
            {
                "criterion_type": "inclusion",
                "raw_text": "Diagnosis of non-insulin-dependent type II diabetes",
                "entity": "insulin",
                "operator": "not_present",
                "value": "insulin"
            },
            {
                "criterion_type": "inclusion",
                "raw_text": "Women over 18 years of age",
                "entity": "UNSTRUCTURED",
                "operator": "llm_eval",
                "value": "Women over 18 years of age"
            },
            {
                "criterion_type": "exclusion",
                "raw_text": "Diagnosis of cancer",
                "entity": "cancer",
                "operator": "present",
                "value": "cancer"
            },
            {
                "criterion_type": "exclusion",
                "raw_text": "Diagnosis of heart failure",
                "entity": "heart_failure",
                "operator": "present",
                "value": "heart failure"
            }
        ],
        "screening_results": [],
        "eligibility_decision": "",
        "eligibility_score": 0.0,
        "outreach_message": "",
        "compliance_approved": False,
        "compliance_issues": [],
        "audit_log": []
    }

    print("\n" + "="*55)
    print("  SCREENER AGENT — TEST RUN")
    print("="*55)
    print(f"  Patient  : {test_state['patient_id']}")
    print(f"  Summary  : {test_state['patient_summary']}")
    print(f"  Trial    : {test_state['trial_id']}")
    print(f"  Criteria : {len(test_state['criteria'])}")
    print("="*55)

    result = screener_agent(test_state)

    print(f"\n  Decision : {result['eligibility_decision']}")
    print(f"  Score    : {result['eligibility_score']}")
    print(f"\n  CRITERION RESULTS:")
    for r in result["screening_results"]:
        icon = "✅" if r.get("met") else "❌"
        print(f"\n  {icon} {r.get('criterion', '')[:60]}")
        print(f"     → {r.get('reasoning', '')[:120]}")

    print(f"\n  AUDIT LOG:")
    for log in result["audit_log"]:
        print(f"  {log}")
    print("="*55)