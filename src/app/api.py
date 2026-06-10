"""
FastAPI backend for the Agentic Clinical Trial Recruitment system.

Exposes the LangGraph orchestrator as HTTP endpoints so the Streamlit UI
(or any other client) can submit a patient + trial and get back the
screening, scoring, outreach, and compliance result.

Run:
    uvicorn app.api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger docs.

>>> THE ONE THING TO ADJUST <<<
Find your real orchestrator entry function in src/agents/orchestrator.py and
match the import + call below (look for the `# ADJUST` markers). Common names:
run_recruitment(), run(), invoke(), build_graph().invoke(state). Also map the
result keys to whatever your RecruitmentState actually contains.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any

app = FastAPI(
    title="Clinical Trial Recruitment API",
    description=(
        "Agentic AI backend: screens patients against trial eligibility, "
        "predicts dropout risk, generates compliant outreach, and enforces "
        "a compliance veto gate via a LangGraph multi-agent orchestrator."
    ),
    version="1.0.0",
)


# ---------- Request / response schemas (Pydantic = automatic validation) ----------
class ScreenRequest(BaseModel):
    patient_id: str
    trial_id: str


class ScreenResponse(BaseModel):
    patient_id: str
    trial_id: str
    eligible: Optional[bool] = None
    eligibility_score: Optional[float] = None
    dropout_risk: Optional[float] = None
    final_score: Optional[float] = None
    outreach_message: Optional[str] = None
    compliance_status: Optional[str] = None
    audit_log: Optional[Any] = None


# ---------- Integration point: import your real orchestrator ----------
# ADJUST this import to match your actual function in src/agents/orchestrator.py
try:
    from src.agents.orchestrator import run_recruitment  # ADJUST function name
    ORCHESTRATOR_AVAILABLE = True
    _import_error = None
except Exception as e:  # keeps the API runnable even before wiring is finished
    ORCHESTRATOR_AVAILABLE = False
    _import_error = str(e)


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Clinical Trial Recruitment API",
        "orchestrator_loaded": ORCHESTRATOR_AVAILABLE,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/screen", response_model=ScreenResponse)
def screen(req: ScreenRequest):
    """Run the full agentic pipeline for one patient against one trial."""
    if not ORCHESTRATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Orchestrator not wired yet: {_import_error}",
        )
    try:
        # ADJUST this call to match your orchestrator's signature.
        # If your orchestrator is a compiled LangGraph, it may be:
        #   result = run_recruitment.invoke({"patient_id": ..., "trial_id": ...})
        result = run_recruitment(patient_id=req.patient_id, trial_id=req.trial_id)

        # result is your RecruitmentState (dict-like). ADJUST keys if yours differ.
        if not isinstance(result, dict):
            result = dict(result)

        return ScreenResponse(
            patient_id=req.patient_id,
            trial_id=req.trial_id,
            eligible=result.get("eligible"),
            eligibility_score=result.get("eligibility_score"),
            dropout_risk=result.get("dropout_risk"),
            final_score=result.get("final_score"),
            outreach_message=result.get("outreach_message"),
            compliance_status=result.get("compliance_status"),
            audit_log=result.get("audit_log"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
