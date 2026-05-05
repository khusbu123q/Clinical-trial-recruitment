# src/pipeline/generate_patients.py
import json
import os
import random
from dataclasses import dataclass, field
from src.utils.config import cfg

random.seed(42)  # reproducible results


@dataclass
class SyntheticPatient:
    patient_id:   str
    age:          int
    sex:          str
    # Labs
    hba1c:        float   # %
    egfr:         float   # mL/min
    bmi:          float   # kg/m2
    systolic_bp:  int     # mmHg
    diastolic_bp: int     # mmHg
    creatinine:   float   # mg/dL
    cholesterol:  int     # mg/dL
    # Conditions
    has_diabetes:      bool
    has_hypertension:  bool
    has_cancer:        bool
    has_heart_failure: bool
    has_liver_disease: bool
    has_ckd:           bool
    # Medications
    on_metformin:   bool
    on_insulin:     bool
    on_ace_inhibitor: bool
    on_statin:      bool
    # Other
    is_smoker:      bool
    is_pregnant:    bool
    # Summary text (mimics clinical note)
    clinical_note:  str = ""

    def to_summary(self) -> str:
        """Short structured summary for model input."""
        conditions = []
        if self.has_diabetes:     conditions.append("Type 2 diabetes")
        if self.has_hypertension: conditions.append("hypertension")
        if self.has_cancer:       conditions.append("active malignancy")
        if self.has_heart_failure:conditions.append("heart failure")
        if self.has_liver_disease:conditions.append("liver disease")
        if self.has_ckd:          conditions.append("CKD")

        medications = []
        if self.on_metformin:    medications.append("metformin")
        if self.on_insulin:      medications.append("insulin")
        if self.on_ace_inhibitor:medications.append("ACE inhibitor")
        if self.on_statin:       medications.append("statin")

        return (
            f"{self.age}yo {self.sex}, "
            f"conditions: {', '.join(conditions) if conditions else 'none'}, "
            f"medications: {', '.join(medications) if medications else 'none'}, "
            f"HbA1c: {self.hba1c}%, eGFR: {self.egfr}, "
            f"BMI: {self.bmi}, BP: {self.systolic_bp}/{self.diastolic_bp}, "
            f"smoker: {self.is_smoker}, pregnant: {self.is_pregnant}"
        )


def _make_clinical_note(p: SyntheticPatient) -> str:
    """Generate a realistic clinical note for the patient."""
    conditions = []
    if p.has_diabetes:      conditions.append("type 2 diabetes mellitus")
    if p.has_hypertension:  conditions.append("hypertension")
    if p.has_heart_failure: conditions.append("congestive heart failure")
    if p.has_liver_disease: conditions.append("chronic liver disease")
    if p.has_ckd:           conditions.append(f"CKD stage {'3' if p.egfr > 30 else '4'}")
    if p.has_cancer:        conditions.append("active malignancy")

    meds = []
    if p.on_metformin:    meds.append("Metformin 1000mg BID")
    if p.on_insulin:      meds.append("Insulin glargine 20 units QHS")
    if p.on_ace_inhibitor:meds.append("Lisinopril 10mg daily")
    if p.on_statin:       meds.append("Atorvastatin 40mg daily")

    note = f"""Patient: {p.patient_id}
Age: {p.age} years, Sex: {p.sex}

Chief Complaint: Routine follow-up

History of Present Illness:
{p.age}-year-old {'male' if p.sex == 'M' else 'female'} presenting for routine follow-up.
{f'Known history of {", ".join(conditions)}.' if conditions else 'No significant past medical history.'}
{'Patient is a current smoker.' if p.is_smoker else 'Non-smoker.'}
{'Patient is currently pregnant.' if p.is_pregnant else ''}

Vital Signs:
Blood Pressure: {p.systolic_bp}/{p.diastolic_bp} mmHg
BMI: {p.bmi} kg/m2

Laboratory Results:
HbA1c: {p.hba1c}%
eGFR: {p.egfr} mL/min/1.73m2
Creatinine: {p.creatinine} mg/dL
Total Cholesterol: {p.cholesterol} mg/dL

Current Medications:
{chr(10).join(f'- {m}' for m in meds) if meds else '- None'}

Assessment:
Patient with {', '.join(conditions) if conditions else 'no active conditions'}.
Labs reviewed and noted.
"""
    return note


def generate_patient(patient_id: str) -> SyntheticPatient:
    """Generate one realistic synthetic patient."""

    # Demographics
    age = random.randint(18, 85)
    sex = random.choice(["M", "F"])

    # Disease prevalence — realistic probabilities
    has_diabetes      = random.random() < 0.35
    has_hypertension  = random.random() < 0.45
    has_cancer        = random.random() < 0.08
    has_heart_failure = random.random() < 0.10
    has_liver_disease = random.random() < 0.07
    has_ckd           = random.random() < 0.15
    is_smoker         = random.random() < 0.18
    is_pregnant       = (sex == "F" and age < 45 and random.random() < 0.05)

    # Labs — influenced by conditions
    hba1c = round(
        random.gauss(8.5, 1.2) if has_diabetes
        else random.gauss(5.4, 0.4), 1
    )
    hba1c = max(4.0, min(14.0, hba1c))

    egfr = round(
        random.gauss(35, 10) if has_ckd
        else random.gauss(75, 15), 1
    )
    egfr = max(5.0, min(120.0, egfr))

    systolic_bp = int(
        random.gauss(155, 15) if has_hypertension
        else random.gauss(118, 12)
    )
    systolic_bp = max(80, min(220, systolic_bp))

    diastolic_bp = int(systolic_bp * random.uniform(0.55, 0.70))

    bmi = round(random.gauss(29, 6), 1)
    bmi = max(15.0, min(55.0, bmi))

    creatinine = round(
        random.gauss(2.2, 0.6) if has_ckd
        else random.gauss(0.95, 0.2), 2
    )
    creatinine = max(0.4, min(8.0, creatinine))

    cholesterol = int(random.gauss(195, 35))
    cholesterol = max(100, min(350, cholesterol))

    # Medications — logical given conditions
    on_metformin    = has_diabetes and not has_ckd and random.random() < 0.75
    on_insulin      = has_diabetes and random.random() < 0.30
    on_ace_inhibitor= (has_hypertension or has_ckd) and random.random() < 0.65
    on_statin       = random.random() < 0.40

    patient = SyntheticPatient(
        patient_id    = patient_id,
        age           = age,
        sex           = sex,
        hba1c         = hba1c,
        egfr          = egfr,
        bmi           = bmi,
        systolic_bp   = systolic_bp,
        diastolic_bp  = diastolic_bp,
        creatinine    = creatinine,
        cholesterol   = cholesterol,
        has_diabetes  = has_diabetes,
        has_hypertension = has_hypertension,
        has_cancer    = has_cancer,
        has_heart_failure = has_heart_failure,
        has_liver_disease = has_liver_disease,
        has_ckd       = has_ckd,
        on_metformin  = on_metformin,
        on_insulin    = on_insulin,
        on_ace_inhibitor = on_ace_inhibitor,
        on_statin     = on_statin,
        is_smoker     = is_smoker,
        is_pregnant   = is_pregnant
    )

    patient.clinical_note = _make_clinical_note(patient)
    return patient


def generate_cohort(n: int = 1000) -> list[SyntheticPatient]:
    """Generate n synthetic patients."""
    print(f"\n👥 Generating {n} synthetic patients...")
    print("-" * 45)

    patients = []
    for i in range(n):
        pid     = f"P{i+1:04d}"
        patient = generate_patient(pid)
        patients.append(patient)

    print(f"✅ Generated {len(patients)} patients")
    return patients


def save_patients(patients: list[SyntheticPatient]):
    """Save patients to JSON."""
    os.makedirs(os.path.join(cfg.data_dir, "synthetic"), exist_ok=True)
    filepath = os.path.join(cfg.data_dir, "synthetic", "patients.json")

    with open(filepath, "w") as f:
        json.dump([p.__dict__ for p in patients], f, indent=2)

    print(f"💾 Saved {len(patients)} patients → {filepath}")
    return filepath


def print_cohort_stats(patients: list[SyntheticPatient]):
    """Print statistics about the generated cohort."""
    n = len(patients)
    print(f"\n{'='*50}")
    print(f"  COHORT STATISTICS (n={n})")
    print(f"{'='*50}")
    print(f"  Demographics:")
    print(f"    Age (mean)     : {sum(p.age for p in patients)/n:.1f} years")
    print(f"    Male           : {sum(1 for p in patients if p.sex=='M')/n*100:.1f}%")
    print(f"    Female         : {sum(1 for p in patients if p.sex=='F')/n*100:.1f}%")
    print(f"\n  Conditions:")
    print(f"    Diabetes       : {sum(1 for p in patients if p.has_diabetes)/n*100:.1f}%")
    print(f"    Hypertension   : {sum(1 for p in patients if p.has_hypertension)/n*100:.1f}%")
    print(f"    CKD            : {sum(1 for p in patients if p.has_ckd)/n*100:.1f}%")
    print(f"    Cancer         : {sum(1 for p in patients if p.has_cancer)/n*100:.1f}%")
    print(f"    Heart failure  : {sum(1 for p in patients if p.has_heart_failure)/n*100:.1f}%")
    print(f"\n  Medications:")
    print(f"    Metformin      : {sum(1 for p in patients if p.on_metformin)/n*100:.1f}%")
    print(f"    Insulin        : {sum(1 for p in patients if p.on_insulin)/n*100:.1f}%")
    print(f"\n  Labs (mean):")
    print(f"    HbA1c          : {sum(p.hba1c for p in patients)/n:.1f}%")
    print(f"    e_sum(p.egfr for p in patients)/n:.1f} mL/min")
    print(f"    BMI            : {sum(p.bmi for p in patients)/n:.1f} kg/m2")
    print(f"\n  Sample patient note:")
    print(f"  {'-'*46}")
    print(patients[0].clinical_note[:400])
    print(f"{'='*50}\n")


if __name__ == "__main__":
    patients = generate_cohort(n=1000)
    print_cohort_stats(patients)
    save_patients(patients)
    print("✅ Patient generation complete")