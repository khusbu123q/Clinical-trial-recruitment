# src/models/eligibility_matcher.py
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from src.utils.config import cfg


@dataclass
class CriterionResult:
    raw_text:   str
    entity:     str
    operator:   str
    value:      str
    unit:       str
    ctype:      str           # "inclusion" or "exclusion"
    met:        bool
    reason:     str
    needs_llm:  bool = False


@dataclass
class EligibilityResult:
    patient_id:         str
    trial_nct_id:       str
    trial_title:        str
    eligible:           bool
    score:              float
    inclusion_results:  list[CriterionResult] = field(default_factory=list)
    exclusion_results:  list[CriterionResult] = field(default_factory=list)
    blocking_reasons:   list[str]             = field(default_factory=list)
    supporting_reasons: list[str]             = field(default_factory=list)
    llm_needed:         bool                  = False

    def summary(self) -> str:
        status = "✅ ELIGIBLE" if self.eligible else "❌ INELIGIBLE"
        lines  = [
            f"{status} | {self.patient_id} × {self.trial_nct_id}",
            f"  Score : {self.score:.2f}",
        ]
        if self.supporting_reasons:
            lines.append(
                f"  ✓  {' | '.join(self.supporting_reasons[:3])}"
            )
        if self.blocking_reasons:
            lines.append(
                f"  ✗  {' | '.join(self.blocking_reasons[:3])}"
            )
        if self.llm_needed:
            lines.append("  ⚠  Some criteria flagged for LLM review")
        return "\n".join(lines)


class RuleBasedEligibilityMatcher:
    """
    Matches extracted patient features against
    parsed trial criteria using deterministic rules.

    Handles:
      - Age range and numeric comparisons (HbA1c, eGFR, BMI …)
      - Presence / absence of diagnoses and medications
      - Flags complex criteria for LLM agent evaluation
    """

    # Map criterion entity strings → patient feature keys
    LAB_ALIASES: dict[str, str] = {
        "hba1c":                    "hba1c",
        "hemoglobin a1c":           "hba1c",
        "glycated hemoglobin":      "hba1c",
        "egfr":                     "egfr",
        "glomerular filtration":    "egfr",
        "creatinine":               "creatinine",
        "bmi":                      "bmi",
        "body mass index":          "bmi",
        "cholesterol":              "cholesterol",
        "total cholesterol":        "cholesterol",
        "glucose":                  "glucose",
        "fasting glucose":          "glucose",
        "systolic blood pressure":  "systolic_bp",
        "systolic bp":              "systolic_bp",
        "diastolic blood pressure": "diastolic_bp",
        "hemoglobin":               "hemoglobin",
        "alt":                      "alt",
        "ast":                      "ast",
        "sodium":                   "sodium",
        "potassium":                "potassium",
        "platelets":                "platelets",
        "wbc":                      "wbc",
    }

    def match(self, patient: dict, trial: dict) -> EligibilityResult:
        """
        Evaluate one patient against one trial.
        patient : dict row from patient_features.json
        trial   : dict row from type_2_diabetes_parsed.json
        """
        inc_results = [
            self._evaluate(patient, c)
            for c in trial.get("inclusion", [])
        ]
        exc_results = [
            self._evaluate(patient, c)
            for c in trial.get("exclusion", [])
        ]

        # Rule: ALL rule-based inclusion criteria must be MET
        #       NO  rule-based exclusion criteria must be MET
        inc_rb = [r for r in inc_results if not r.needs_llm]
        exc_rb = [r for r in exc_results if not r.needs_llm]

        all_inc_met  = all(r.met for r in inc_rb)
        any_exc_met  = any(r.met for r in exc_rb)
        has_llm      = any(
            r.needs_llm for r in inc_results + exc_results
        )

        eligible = all_inc_met and not any_exc_met

        # Score = fraction of inclusion criteria met (rule-based only)
        score = (
            sum(1 for r in inc_rb if r.met) / max(len(inc_rb), 1)
        )

        supporting = [r.reason for r in inc_rb if r.met]
        blocking   = (
            [r.reason for r in inc_rb if not r.met] +
            [r.reason for r in exc_rb if r.met]
        )

        return EligibilityResult(
            patient_id         = patient.get("patient_id", "?"),
            trial_nct_id       = trial.get("nct_id", "?"),
            trial_title        = trial.get("title", "?"),
            eligible           = eligible,
            score              = score,
            inclusion_results  = inc_results,
            exclusion_results  = exc_results,
            blocking_reasons   = blocking,
            supporting_reasons = supporting,
            llm_needed         = has_llm,
        )

    # ── Criterion evaluation ──────────────────────────────────

    def _evaluate(
        self, patient: dict, criterion: dict
    ) -> CriterionResult:
        """Route one criterion to the right evaluation method."""

        # Unpack criterion dict (comes from parsed JSON)
        entity   = str(criterion.get("entity",   "")).lower().strip()
        operator = str(criterion.get("operator", "")).lower().strip()
        value    = str(criterion.get("value",    "")).strip()
        unit     = str(criterion.get("unit",     "")).strip()
        ctype    = str(criterion.get("criterion_type", "inclusion"))
        raw      = str(criterion.get("raw_text",  ""))

        def _result(met: bool, reason: str, llm: bool = False):
            return CriterionResult(
                raw_text  = raw,
                entity    = entity,
                operator  = operator,
                value     = value,
                unit      = unit,
                ctype     = ctype,
                met       = met,
                reason    = reason,
                needs_llm = llm,
            )

        # ── Flag LLM criteria immediately ─────────────────────
        if operator == "llm_eval":
            return _result(
                met    = True,   # optimistic — LLM decides
                reason = f"LLM needed: {raw[:70]}",
                llm    = True,
            )

        # ── Age ───────────────────────────────────────────────
        if entity == "age":
            return self._check_age(patient, operator, value, _result)

        # ── Lab / vital numeric check ──────────────────────────
        for alias, key in self.LAB_ALIASES.items():
            if alias in entity:
                patient_val = (
                    patient.get("lab_values",  {}).get(key) or
                    patient.get("vitals",       {}).get(key)
                )
                if patient_val is None:
                    return _result(
                        True,
                        f"{key} not in patient record",
                        llm=True,
                    )
                try:
                    threshold = float(value)
                except ValueError:
                    return _result(True, f"Cannot parse threshold '{value}'", llm=True)
                return self._check_numeric(
                    patient_val, operator, threshold, key, unit, _result
                )

        # ── Presence / absence ────────────────────────────────
        if operator in ("present", "not_present"):
            return self._check_presence(patient, entity, value, operator, _result)

        # ── Between (non-age) ─────────────────────────────────
        if operator == "between":
            for alias, key in self.LAB_ALIASES.items():
                if alias in entity:
                    patient_val = (
                        patient.get("lab_values", {}).get(key) or
                        patient.get("vitals",      {}).get(key)
                    )
                    if patient_val is None:
                        return _result(True, f"{key} not found", llm=True)
                    try:
                        lo_str, hi_str = value.split("-")
                        lo, hi = float(lo_str), float(hi_str)
                        met = lo <= patient_val <= hi
                        return _result(
                            met,
                            f"{key} {patient_val} "
                            f"{'within' if met else 'outside'} "
                            f"{lo}–{hi} {unit}"
                        )
                    except Exception:
                        return _result(True, f"Range parse error: {value}", llm=True)

        # ── Cannot evaluate — send to LLM ─────────────────────
        return _result(
            True,
            f"Unhandled criterion: {raw[:60]}",
            llm=True,
        )

    def _check_age(
        self, patient: dict, operator: str, value: str, _result
    ) -> CriterionResult:
        age = patient.get("age")
        if age is None:
            return _result(True, "Age not in record", llm=True)

        if operator == "between":
            try:
                lo, hi = value.split("-")
                met = int(lo) <= age <= int(hi)
                return _result(
                    met,
                    f"Age {age} {'within' if met else 'outside'} "
                    f"{lo}–{hi} years"
                )
            except Exception:
                return _result(True, f"Age range parse error: {value}", llm=True)

        try:
            threshold = float(value)
        except ValueError:
            return _result(True, f"Cannot parse age threshold '{value}'", llm=True)

        return self._check_numeric(age, operator, threshold, "age", "years", _result)

    def _check_numeric(
        self,
        patient_val: float,
        operator:    str,
        threshold:   float,
        label:       str,
        unit:        str,
        _result,
    ) -> CriterionResult:
        ops = {
            ">":  patient_val >  threshold,
            "<":  patient_val <  threshold,
            ">=": patient_val >= threshold,
            "<=": patient_val <= threshold,
            "==": abs(patient_val - threshold) < 0.01,
        }
        met = ops.get(operator, False)
        return _result(
            met,
            f"{label} {patient_val} "
            f"{'meets' if met else 'fails'} "
            f"{operator} {threshold} {unit}".strip()
        )

    def _check_presence(
        self,
        patient:  dict,
        entity:   str,
        value:    str,
        operator: str,
        _result,
    ) -> CriterionResult:
        search_term = value.lower()
        diagnoses   = [d.lower() for d in patient.get("diagnoses",  [])]
        medications = [m.lower() for m in patient.get("medications", [])]

        found = (
            any(search_term in d or d in search_term for d in diagnoses) or
            any(search_term in m or m in search_term for m in medications)
        )

        if operator == "present":
            return _result(
                found,
                f"{search_term} "
                f"{'found ✓' if found else 'not found ✗'} in record"
            )
        else:  # not_present
            return _result(
                not found,
                f"{search_term} "
                f"{'absent ✓' if not found else 'present — exclusion triggered ✗'}"
            )


# ── Batch runner ──────────────────────────────────────────────

def run_matching(
    n_patients: int = 50,
    n_trials:   int = 3,
) -> list[EligibilityResult]:
    BASE = "/Users/khusbuagarwal/Desktop/clinical_trial_recruitment/data"
    patients_path = os.path.join(BASE, "processed", "patient_features.json")
    trials_path = os.path.join(BASE, "processed", "type_2_diabetes_parsed.json")
    if not os.path.exists(patients_path):
        print("❌ patient_features.json not found — run ner_pipeline.py first")
        return []
    if not os.path.exists(trials_path):
        print("❌ type_2_diabetes_parsed.json not found — run criteria_parser.py first")
        return []

    with open(patients_path) as f:
        patients = json.load(f)[:n_patients]
    with open(trials_path) as f:
        trials = json.load(f)[:n_trials]

    matcher = RuleBasedEligibilityMatcher()
    results = []

    print(f"\n🔍 Matching {len(patients)} patients × {len(trials)} trials")
    print(f"   = {len(patients) * len(trials)} total evaluations")
    print("-" * 50)

    for trial in trials:
        elig_count = 0
        for patient in patients:
            r = matcher.match(patient, trial)
            results.append(r)
            if r.eligible:
                elig_count += 1

        print(
            f"  📋 {trial['nct_id']} — "
            f"{trial['title'][:45]}..."
            f"\n     Eligible: {elig_count}/{len(patients)}"
        )

    return results

def save_results(results: list[EligibilityResult]):
    BASE = "/Users/khusbuagarwal/Desktop/clinical_trial_recruitment/data"
    out_path = os.path.join(BASE, "processed", "matching_results.json")
    with open(out_path, "w") as f:
        json.dump(
            [{
                "patient_id":   r.patient_id,
                "trial_nct_id": r.trial_nct_id,
                "trial_title":  r.trial_title[:60],
                "eligible":     r.eligible,
                "score":        round(r.score, 3),
                "blocking":     r.blocking_reasons[:5],
                "supporting":   r.supporting_reasons[:5],
                "llm_needed":   r.llm_needed,
            } for r in results],
            f, indent=2,
        )
    print(f"💾 Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_matching(n_patients=50, n_trials=3)

    if not results:
        exit(1)

    # ── Per-trial summary ──────────────────────────────────────
    from collections import defaultdict
    by_trial: dict[str, list[EligibilityResult]] = defaultdict(list)
    for r in results:
        by_trial[r.trial_nct_id].append(r)

    for nct_id, trial_results in by_trial.items():
        eligible   = [r for r in trial_results if r.eligible]
        ineligible = [r for r in trial_results if not r.eligible]
        llm_needed = [r for r in trial_results if r.llm_needed]

        print(f"\n{'='*52}")
        print(f"  TRIAL: {nct_id}")
        print(f"  {trial_results[0].trial_title[:55]}...")
        print(f"{'='*52}")
        print(f"  Eligible     : {len(eligible)}/{len(trial_results)}")
        print(f"  Ineligible   : {len(ineligible)}/{len(trial_results)}")
        print(f"  LLM flagged  : {len(llm_needed)}/{len(trial_results)}")

        if eligible:
            print(f"\n  TOP ELIGIBLE PATIENTS:")
            # Sort by score descending
            for r in sorted(eligible, key=lambda x: x.score, reverse=True)[:3]:
                print(f"\n  {r.summary()}")

        if ineligible:
            print(f"\n  SAMPLE INELIGIBLE (why blocked):")
            for r in ineligible[:2]:
                print(f"\n  {r.summary()}")

    # ── Overall summary ────────────────────────────────────────
    total    = len(results)
    elig     = sum(1 for r in results if r.eligible)
    llm_flag = sum(1 for r in results if r.llm_needed)
    avg_score = sum(r.score for r in results) / max(total, 1)

    print(f"\n{'='*52}")
    print(f"  OVERALL MATCHING SUMMARY")
    print(f"{'='*52}")
    print(f"  Total evaluations  : {total}")
    print(f"  Eligible           : {elig}  ({elig/total*100:.1f}%)")
    print(f"  Ineligible         : {total-elig}  ({(total-elig)/total*100:.1f}%)")
    print(f"  Need LLM review    : {llm_flag}  ({llm_flag/total*100:.1f}%)")
    print(f"  Avg match score    : {avg_score:.3f}")
    print(f"{'='*52}\n")

    save_results(results)
    print("✅ Eligibility matching complete")