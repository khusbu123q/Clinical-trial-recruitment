# src/pipeline/ner_pipeline.py
import json
import os
import re
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional
from src.utils.config import cfg


@dataclass
class ExtractedFeatures:
    patient_id:  str
    age:         Optional[int]  = None
    sex:         Optional[str]  = None
    diagnoses:   list[str]      = field(default_factory=list)
    medications: list[str]      = field(default_factory=list)
    lab_values:  dict           = field(default_factory=dict)
    vitals:      dict           = field(default_factory=dict)
    negated:     list[str]      = field(default_factory=list)
    summary:     str            = ""

    def to_model_input(self) -> str:
        """Format as single string for ClinicalBERT input."""
        parts = []
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.sex:
            parts.append(f"Sex: {self.sex}")
        if self.diagnoses:
            parts.append(f"Diagnoses: {', '.join(self.diagnoses)}")
        if self.medications:
            parts.append(f"Medications: {', '.join(self.medications)}")
        if self.lab_values:
            labs = ", ".join(f"{k}: {v}" for k, v in self.lab_values.items())
            parts.append(f"Labs: {labs}")
        if self.vitals:
            vit = ", ".join(f"{k}: {v}" for k, v in self.vitals.items())
            parts.append(f"Vitals: {vit}")
        if self.negated:
            parts.append(f"Absent: {', '.join(self.negated[:5])}")
        return " | ".join(parts)


class ClinicalNERPipeline:
    """
    Rule-based clinical NER pipeline.
    Extracts diagnoses, medications, labs, vitals
    from free-text clinical notes using regex patterns.
    No GPU needed — runs on CPU instantly.
    """

    # ── Disease patterns ──────────────────────────────────────
    DISEASE_PATTERNS = {
        "type_2_diabetes":  r"type\s*2\s*diabet|t2dm|t2d(?!\w)|"
                            r"non.insulin.dependent diabet",
        "type_1_diabetes":  r"type\s*1\s*diabet|t1dm|"
                            r"insulin.dependent diabet",
        "hypertension":     r"\bhypertension\b|high blood pressure|\bhtn\b",
        "ckd":              r"chronic kidney disease|\bckd\b|"
                            r"renal insufficiency|renal failure",
        "heart_failure":    r"heart failure|\bchf\b|cardiac failure|"
                            r"congestive heart",
        "cancer":           r"\bcancer\b|\bmalignancy\b|carcinoma|"
                            r"\btumor\b|\blymphoma\b|\bleukemia\b",
        "liver_disease":    r"liver disease|\bhepatitis\b|"
                            r"\bcirrhosis\b|\bnafld\b|\bnash\b",
        "stroke":           r"\bstroke\b|cerebrovascular|\btia\b|"
                            r"transient ischemic",
        "myocardial_infarction": r"myocardial infarction|heart attack|\bmi\b(?!\s*[:\s]\d)",
        "copd":             r"\bcopd\b|chronic obstructive|\bemphysema\b",
        "obesity":          r"\bobesity\b|\bobese\b",
        "depression":       r"\bdepression\b|depressive disorder|\bmdd\b",
        "hypothyroidism":   r"hypothyroid|underactive thyroid",
        "atrial_fibrillation": r"atrial fibrillation|\bafib\b|\baf\b(?!\w)",
        "asthma":           r"\basthma\b|reactive airway",
    }

    # ── Medication patterns ───────────────────────────────────
    MED_PATTERNS = {
        "metformin":        r"\bmetformin\b|\bglucophage\b",
        "insulin":          r"\binsulin\b",
        "lisinopril":       r"\blisinopril\b|\bzestril\b|\bprinivil\b",
        "atorvastatin":     r"\batorvastatin\b|\blipitor\b",
        "amlodipine":       r"\bamlodipine\b|\bnorvasc\b",
        "aspirin":          r"\baspirin\b|\basa\b(?!\w)",
        "warfarin":         r"\bwarfarin\b|\bcoumadin\b",
        "glipizide":        r"\bglipizide\b|\bglucontrol\b",
        "sitagliptin":      r"\bsitagliptin\b|\bjanuvia\b",
        "empagliflozin":    r"\bempagliflozin\b|\bjardiance\b",
        "liraglutide":      r"\bliraglutide\b|\bvictoza\b",
        "losartan":         r"\blosartan\b|\bcozaar\b",
        "chemotherapy":     r"\bchemotherapy\b|\bchemo\b|\bcytotoxic\b",
        "corticosteroid":   r"\bprednisone\b|\bdexamethasone\b|"
                            r"\bmethylprednisolone\b|\bcorticosteroid\b",
        "ace_inhibitor":    r"ace inhibitor|angiotensin.converting enzyme",
        "beta_blocker":     r"\bmetoprolol\b|\bbisoprolol\b|\bcarvedilol\b|"
                            r"beta.blocker",
        "diuretic":         r"\bfurosemide\b|\bhydrochlorothiazide\b|"
                            r"\bhctz\b|\bdiuretic\b",
    }

    # ── Lab value patterns ────────────────────────────────────
    # Each pattern has ONE capture group for the numeric value
    LAB_PATTERNS = {
        "hba1c":        r"hba1c\s*[:\s]\s*(\d+\.?\d*)\s*%?",
        "egfr":         r"egfr\s*[:\s]\s*(\d+\.?\d*)",
        "creatinine":   r"creatinine\s*[:\s]\s*(\d+\.?\d*)",
        "cholesterol":  r"(?:total\s*)?cholesterol\s*[:\s]\s*(\d+\.?\d*)",
        "glucose":      r"(?:fasting\s*)?glucose\s*[:\s]\s*(\d+\.?\d*)",
        "bmi":          r"\bbmi\s*[:\s]\s*(\d+\.?\d*)",
        "hemoglobin":   r"\bhemoglobin\s*[:\s]\s*(\d+\.?\d*)",
        "wbc":          r"\bwbc\s*[:\s]\s*(\d+\.?\d*)",
        "platelets":    r"\bplatelets?\s*[:\s]\s*(\d+)",
        "alt":          r"\balt\s*[:\s]\s*(\d+\.?\d*)",
        "ast":          r"\bast\s*[:\s]\s*(\d+\.?\d*)",
        "sodium":       r"\bsodium\s*[:\s]\s*(\d+\.?\d*)",
        "potassium":    r"\bpotassium\s*[:\s]\s*(\d+\.?\d*)",
    }

    # ── Vital sign patterns ───────────────────────────────────
    VITAL_PATTERNS = {
        "systolic_bp":  r"(?:blood pressure|bp)\s*[:\s]\s*(\d+)/\d+",
        "diastolic_bp": r"(?:blood pressure|bp)\s*[:\s]\s*\d+/(\d+)",
        "heart_rate":   r"(?:heart rate|hr|pulse)\s*[:\s]\s*(\d+)",
        "temperature":  r"(?:temperature|temp)\s*[:\s]\s*(\d+\.?\d*)",
        "weight_kg":    r"\bweight\s*[:\s]\s*(\d+\.?\d*)\s*kg",
        "height_cm":    r"\bheight\s*[:\s]\s*(\d+\.?\d*)\s*cm",
        "spo2":         r"(?:spo2|oxygen saturation)\s*[:\s]\s*(\d+\.?\d*)\s*%?",
    }

    # ── Negation context words ────────────────────────────────
    NEGATION_WORDS = [
        "no ", "not ", "without", "denies", "negative for",
        "absence of", "never ", "no history of", "no evidence of",
        "rules out", "ruled out", "unremarkable for"
    ]

    def extract(self, note: str, patient_id: str) -> ExtractedFeatures:
        """
        Main extraction method.
        Takes a clinical note string, returns ExtractedFeatures.
        """
        features   = ExtractedFeatures(patient_id=patient_id)
        note_lower = note.lower()

        # 1. Extract age
        features.age = self._extract_age(note_lower)

        # 2. Extract sex
        features.sex = self._extract_sex(note_lower)

        # 3. Extract negated terms (must do before positive extraction)
        features.negated = self._extract_negated(note_lower)

        # 4. Extract diagnoses (skip negated ones)
        features.diagnoses = self._extract_entities(
            note_lower, self.DISEASE_PATTERNS, features.negated
        )

        # 5. Extract medications (skip negated ones)
        features.medications = self._extract_entities(
            note_lower, self.MED_PATTERNS, features.negated
        )

        # 6. Extract lab values
        features.lab_values = self._extract_numerics(
            note_lower, self.LAB_PATTERNS
        )

        # 7. Extract vitals
        features.vitals = self._extract_numerics(
            note_lower, self.VITAL_PATTERNS
        )

        # 8. Build model input string
        features.summary = features.to_model_input()

        return features

    # ── Private extraction helpers ────────────────────────────

    def _extract_age(self, text: str) -> Optional[int]:
        patterns = [
            r"(\d+)[\s-]*(?:year|yr)[\s-]*old",
            r"age[d]?\s*[:\s]\s*(\d+)",
            r"(\d+)[\s-]*y/?o\b",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                age = int(m.group(1))
                if 0 < age < 120:
                    return age
        return None

    def _extract_sex(self, text: str) -> Optional[str]:
        if re.search(r"\bmale\b|\bman\b|\bgentleman\b", text):
            return "M"
        if re.search(r"\bfemale\b|\bwoman\b|\blady\b", text):
            return "F"
        return None

    def _extract_negated(self, text: str) -> list[str]:
        """Find all terms that appear in a negation context."""
        negated = set()
        neg_patterns = [
            r"no\s+(?:history\s+of\s+|evidence\s+of\s+)?(\w+(?:\s+\w+){0,2})",
            r"without\s+(\w+(?:\s+\w+){0,1})",
            r"denies\s+(\w+(?:\s+\w+){0,1})",
            r"negative\s+for\s+(\w+(?:\s+\w+){0,1})",
            r"absence\s+of\s+(\w+(?:\s+\w+){0,1})",
        ]
        for p in neg_patterns:
            for m in re.finditer(p, text):
                negated.add(m.group(1).strip())
        return list(negated)

    def _extract_entities(
        self,
        text:    str,
        patterns: dict[str, str],
        negated:  list[str]
    ) -> list[str]:
        """Extract entities, skipping those in negation context."""
        found = []
        for name, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if not self._in_negation_context(text, pattern):
                    found.append(name)
        return found

    def _extract_numerics(
        self,
        text:    str,
        patterns: dict[str, str]
    ) -> dict[str, float]:
        """Extract numeric lab/vital values."""
        values = {}
        for name, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    values[name] = float(m.group(1))
                except (ValueError, IndexError):
                    pass
        return values

    def _in_negation_context(self, text: str, pattern: str) -> bool:
        """Check if pattern match is preceded by negation words."""
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return False
        start   = max(0, m.start() - 60)
        context = text[start:m.start()]
        return any(neg in context for neg in self.NEGATION_WORDS)


# ── Pipeline runner ───────────────────────────────────────────

def run_ner_pipeline(patients: list[dict]) -> list[ExtractedFeatures]:
    """Run NER pipeline on a list of patient dicts."""
    pipeline = ClinicalNERPipeline()
    results  = []

    print(f"\n🔬 Running NER pipeline on {len(patients)} patients...")
    print("-" * 45)

    for i, p in enumerate(patients):
        note     = p.get("clinical_note", "")
        pid      = p.get("patient_id", f"P{i+1:04d}")
        features = pipeline.extract(note, pid)
        results.append(features)

        if (i + 1) % 200 == 0:
            print(f"  ✓ Processed {i+1}/{len(patients)}...")

    print(f"✅ NER complete — {len(results)} patients processed")
    return results


def print_ner_stats(results: list[ExtractedFeatures]):
    """Print extraction statistics."""
    n = len(results)

    has_age  = sum(1 for r in results if r.age  is not None)
    has_sex  = sum(1 for r in results if r.sex  is not None)
    has_diag = sum(1 for r in results if r.diagnoses)
    has_meds = sum(1 for r in results if r.medications)
    has_labs = sum(1 for r in results if r.lab_values)
    has_vit  = sum(1 for r in results if r.vitals)

    print(f"\n{'='*52}")
    print(f"  NER EXTRACTION STATISTICS  (n={n})")
    print(f"{'='*52}")
    print(f"  Field extraction rate:")
    print(f"    Age          : {has_age /n*100:5.1f}%")
    print(f"    Sex          : {has_sex /n*100:5.1f}%")
    print(f"    Diagnoses    : {has_diag/n*100:5.1f}%")
    print(f"    Medications  : {has_meds/n*100:5.1f}%")
    print(f"    Lab values   : {has_labs/n*100:5.1f}%")
    print(f"    Vitals       : {has_vit /n*100:5.1f}%")

    all_dx   = [dx for r in results for dx in r.diagnoses]
    all_meds = [m  for r in results for m  in r.medications]

    print(f"\n  Top diagnoses extracted:")
    for dx, cnt in Counter(all_dx).most_common(6):
        print(f"    {dx:<30} {cnt:4d}  ({cnt/n*100:.1f}%)")

    print(f"\n  Top medications extracted:")
    for med, cnt in Counter(all_meds).most_common(6):
        print(f"    {med:<30} {cnt:4d}  ({cnt/n*100:.1f}%)")

    print(f"\n  Sample — P0001:")
    p = results[0]
    print(f"    Age         : {p.age}")
    print(f"    Sex         : {p.sex}")
    print(f"    Diagnoses   : {p.diagnoses}")
    print(f"    Medications : {p.medications}")
    print(f"    Labs        : {p.lab_values}")
    print(f"    Vitals      : {p.vitals}")
    print(f"\n  Model input string:")
    print(f"    {p.summary[:220]}")
    print(f"{'='*52}\n")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    filepath = os.path.join(cfg.data_dir, "synthetic", "patients.json")

    if not os.path.exists(filepath):
        print("❌ patients.json not found — run generate_patients.py first")
        exit(1)

    with open(filepath) as f:
        patients = json.load(f)

    results = run_ner_pipeline(patients)
    print_ner_stats(results)

    # Save extracted features
    out_path = os.path.join(
        cfg.data_dir, "processed", "patient_features.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    print(f"💾 Saved → {out_path}")
    print(f"✅ NER pipeline complete")