# src/pipeline/criteria_parser.py
import re
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from src.utils.config import cfg


@dataclass
class ParsedCriterion:
    raw_text:       str
    criterion_type: str        # "inclusion" or "exclusion"
    entity:         str        # what is being checked e.g. "HbA1c", "age"
    operator:       str        # ">", "<", ">=", "<=", "==", "not_present", "present", "llm_eval"
    value:          str        # "7.5", "18", "insulin"
    unit:           str = ""   # "%", "years", "mg/dL"
    confidence:     str = "rule_based"  # "rule_based" or "needs_llm"


class CriteriaParser:

    # Numeric patterns — catches things like "HbA1c >= 7.5%", "age > 18 years"
    NUMERIC_PATTERN = re.compile(
        r"([\w\s\-/]+?)\s*(>=|<=|>|<|=|of|between)\s*(\d+\.?\d*)\s*"
        r"(%|mg/dL|mmol/L|years?|kg/m2|mL/min|IU/L|g/dL|ng/mL)?",
        re.IGNORECASE
    )

    # Age range pattern — catches "18 to 75 years", "aged 18-70"
    AGE_RANGE_PATTERN = re.compile(
        r"age[d]?\s*(?:of\s*)?(\d+)\s*(?:to|-|–)\s*(\d+)\s*years?",
        re.IGNORECASE
    )

    # Presence/absence patterns
    NEGATION_WORDS   = ["no ", "not ", "without", "absence of", "free of",
                        "no history of", "never ", "non-"]
    PRESENCE_WORDS   = ["history of", "diagnosis of", "presence of",
                        "known ", "confirmed ", "documented "]

    # Key medical entities to watch for
    MEDICAL_ENTITIES = [
        "insulin", "metformin", "chemotherapy", "pregnancy", "cancer",
        "malignancy", "hiv", "hepatitis", "renal failure", "liver disease",
        "heart failure", "stroke", "surgery", "diabetes", "hypertension",
        "smoking", "alcohol", "pregnant", "breastfeeding", "allergy"
    ]

    def parse_trial(self, trial: dict) -> dict:
        """Parse both inclusion and exclusion criteria from a trial dict."""
        return {
            "nct_id":    trial.get("nct_id", ""),
            "title":     trial.get("title", ""),
            "inclusion": self.parse_block(
                trial.get("inclusion_criteria", ""), "inclusion"
            ),
            "exclusion": self.parse_block(
                trial.get("exclusion_criteria", ""), "exclusion"
            )
        }

    def parse_block(self, text: str, ctype: str) -> list[ParsedCriterion]:
        """Parse a full criteria block into individual criteria."""
        if not text or not text.strip():
            return []

        lines = self._split_into_lines(text)
        criteria = []

        for line in lines:
            line = line.strip()
            if len(line) < 10:   # skip very short lines
                continue
            criterion = self._parse_single(line, ctype)
            if criterion:
                criteria.append(criterion)

        return criteria

    def _split_into_lines(self, text: str) -> list[str]:
        """Split criteria block into individual criterion lines."""
        # Remove common headers
        for header in ["Inclusion Criteria:", "Exclusion Criteria:",
                       "Inclusion:", "Exclusion:"]:
            text = text.replace(header, "").strip()

        # Split on numbered lists (1. 2. 3.) or bullets (* - •)
        lines = re.split(r'\n\s*(?:\d+[.)]\s*|[-*•]\s*)', text)

        # Also split on semicolons that separate criteria
        result = []
        for line in lines:
            if ";" in line and len(line) > 80:
                result.extend(line.split(";"))
            else:
                result.append(line)

        return [l.strip() for l in result if l.strip()]

    def _parse_single(self, text: str, ctype: str) -> Optional[ParsedCriterion]:
        """Try to parse one criterion line into structured form."""
        text_clean = text.strip().rstrip(".")

        # 1. Try age range pattern first
        age_match = self.AGE_RANGE_PATTERN.search(text_clean)
        if age_match:
            min_age = age_match.group(1)
            max_age = age_match.group(2)
            return ParsedCriterion(
                raw_text       = text_clean,
                criterion_type = ctype,
                entity         = "age",
                operator       = "between",
                value          = f"{min_age}-{max_age}",
                unit           = "years",
                confidence     = "rule_based"
            )

        # 2. Try numeric comparison pattern
        num_match = self.NUMERIC_PATTERN.search(text_clean)
        if num_match:
            entity   = num_match.group(1).strip().lower()
            operator = self._normalize_operator(num_match.group(2))
            value    = num_match.group(3)
            unit     = num_match.group(4) or ""

            # Filter out garbage matches
            if len(entity) > 40 or len(entity) < 2:
                pass
            else:
                return ParsedCriterion(
                    raw_text       = text_clean,
                    criterion_type = ctype,
                    entity         = entity,
                    operator       = operator,
                    value          = value,
                    unit           = unit,
                    confidence     = "rule_based"
                )

        # 3. Try presence/absence of medical entity
        text_lower = text_clean.lower()
        for entity in self.MEDICAL_ENTITIES:
            if entity in text_lower:
                is_negated = any(neg in text_lower for neg in self.NEGATION_WORDS)
                return ParsedCriterion(
                    raw_text       = text_clean,
                    criterion_type = ctype,
                    entity         = entity,
                    operator       = "not_present" if is_negated else "present",
                    value          = entity,
                    unit           = "",
                    confidence     = "rule_based"
                )

        # 4. Cannot be parsed by rules — needs LLM evaluation
        return ParsedCriterion(
            raw_text       = text_clean,
            criterion_type = ctype,
            entity         = "UNSTRUCTURED",
            operator       = "llm_eval",
            value          = text_clean,
            unit           = "",
            confidence     = "needs_llm"
        )

    def _normalize_operator(self, op: str) -> str:
        mapping = {
            "of":      "==",
            "=":       "==",
            ">=":      ">=",
            "<=":      "<=",
            ">":       ">",
            "<":       "<",
            "between": "between"
        }
        return mapping.get(op.lower().strip(), "==")


def parse_all_trials(condition: str) -> list[dict]:
    """Load saved trial JSON and parse all criteria."""
    filename = condition.lower().replace(" ", "_") + "_trials.json"
    filepath = os.path.join(cfg.data_dir, "raw", filename)

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print(f"   Run fetch_trials.py first")
        return []

    with open(filepath) as f:
        trials = json.load(f)

    parser  = CriteriaParser()
    results = []

    for trial in trials:
        parsed = parser.parse_trial(trial)
        results.append(parsed)

    return results


def print_stats(results: list[dict]):
    """Print parsing statistics."""
    total_inc       = sum(len(r["inclusion"]) for r in results)
    total_exc       = sum(len(r["exclusion"]) for r in results)
    rule_based      = sum(
        1 for r in results
        for c in r["inclusion"] + r["exclusion"]
        if c.confidence == "rule_based"
    )
    needs_llm       = sum(
        1 for r in results
        for c in r["inclusion"] + r["exclusion"]
        if c.confidence == "needs_llm"
    )
    total_criteria  = rule_based + needs_llm

    print(f"\n{'='*50}")
    print(f"  PARSING STATISTICS")
    print(f"{'='*50}")
    print(f"  Trials parsed       : {len(results)}")
    print(f"  Inclusion criteria  : {total_inc}")
    print(f"  Exclusion criteria  : {total_exc}")
    print(f"  Total criteria      : {total_criteria}")
    print(f"  Rule-based parsed   : {rule_based} "
          f"({rule_based/max(total_criteria,1)*100:.1f}%)")
    print(f"  Needs LLM eval      : {needs_llm} "
          f"({needs_llm/max(total_criteria,1)*100:.1f}%)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    condition = "type 2 diabetes"
    print(f"\n📋 Parsing criteria for: {condition}")

    results = parse_all_trials(condition)

    if not results:
        exit(1)

    # Show detailed parse of first 3 trials
    parser = CriteriaParser()

    for i, result in enumerate(results[:3]):
        print(f"\n{'='*50}")
        print(f"Trial {i+1}: {result['nct_id']}")
        print(f"  {result['title'][:60]}...")
        print(f"{'='*50}")

        print(f"\n  INCLUSION CRITERIA ({len(result['inclusion'])} parsed):")
        for c in result["inclusion"]:
            icon = "🔵" if c.confidence == "rule_based" else "🟡"
            print(f"  {icon} [{c.entity}] {c.operator} {c.value} {c.unit}")
            print(f"     raw: {c.raw_text[:80]}")

        print(f"\n  EXCLUSION CRITERIA ({len(result['exclusion'])} parsed):")
        for c in result["exclusion"]:
            icon = "🔵" if c.confidence == "rule_based" else "🟡"
            print(f"  {icon} [{c.entity}] {c.operator} {c.value} {c.unit}")
            print(f"     raw: {c.raw_text[:80]}")

    # Overall stats
    print_stats(results)

    # Save parsed results
    out_path = os.path.join(cfg.data_dir, "processed",
                            "type_2_diabetes_parsed.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(
            [{
                "nct_id": r["nct_id"],
                "title":  r["title"],
                "inclusion": [c.__dict__ for c in r["inclusion"]],
                "exclusion": [c.__dict__ for c in r["exclusion"]]
            } for r in results],
            f, indent=2
        )

    print(f"💾 Saved parsed criteria → {out_path}")