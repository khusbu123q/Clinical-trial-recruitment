# src/pipeline/fetch_trials.py
import requests
import json
import os
import time
from dataclasses import dataclass
from typing import Optional
from src.utils.config import cfg


@dataclass
class Trial:
    nct_id: str
    title: str
    condition: str
    phase: str
    status: str
    inclusion_criteria: str
    exclusion_criteria: str
    min_age: str
    max_age: str
    sex: str
    healthy_volunteers: str
    sponsor: str
    raw_criteria: str


class ClinicalTrialsFetcher:
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self):
        os.makedirs(os.path.join(cfg.data_dir, "raw"), exist_ok=True)

    def fetch(self, condition: str, n: int = 50) -> list[Trial]:
        print(f"\n🔍 Fetching {n} trials for: '{condition}'")
        print("-" * 45)

        params = {
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
            "pageSize": min(n, 100),
            "format": "json"
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API call failed: {e}")
            return []

        studies = data.get("studies", [])
        print(f"✅ Retrieved {len(studies)} studies from API")

        trials = []
        for s in studies:
            trial = self._parse_study(s)
            if trial:
                trials.append(trial)

        print(f"✅ Successfully parsed {len(trials)} trials")
        return trials

    def _parse_study(self, study: dict) -> Optional[Trial]:
        try:
            protocol    = study.get("protocolSection", {})
            id_mod      = protocol.get("identificationModule", {})
            status_mod  = protocol.get("statusModule", {})
            design_mod  = protocol.get("designModule", {})
            elig_mod    = protocol.get("eligibilityModule", {})
            cond_mod    = protocol.get("conditionsModule", {})
            sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})

            nct_id = id_mod.get("nctId", "")
            title  = id_mod.get("briefTitle", "")
            if not nct_id or not title:
                return None

            raw_criteria        = elig_mod.get("eligibilityCriteria", "")
            inclusion, exclusion = self._split_criteria(raw_criteria)

            conditions = cond_mod.get("conditions", [])
            condition  = conditions[0] if conditions else ""

            phases = design_mod.get("phases", [])
            phase  = phases[0] if phases else "N/A"

            sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")

            return Trial(
                nct_id             = nct_id,
                title              = title,
                condition          = condition,
                phase              = phase,
                status             = status_mod.get("overallStatus", ""),
                inclusion_criteria = inclusion,
                exclusion_criteria = exclusion,
                raw_criteria       = raw_criteria,
                min_age            = elig_mod.get("minimumAge", "N/A"),
                max_age            = elig_mod.get("maximumAge", "N/A"),
                sex                = elig_mod.get("sex", "ALL"),
                healthy_volunteers = elig_mod.get("healthyVolunteers", "No"),
                sponsor            = sponsor
            )
        except Exception as e:
            print(f"  ⚠️  Could not parse study: {e}")
            return None

    def _split_criteria(self, raw: str) -> tuple[str, str]:
        if not raw:
            return "", ""

        raw_lower = raw.lower()
        split_idx = -1

        for marker in ["exclusion criteria", "exclusion criterion", "exclude "]:
            idx = raw_lower.find(marker)
            if idx != -1:
                split_idx = idx
                break

        if split_idx == -1:
            return raw.strip(), ""

        inclusion = raw[:split_idx].strip()
        exclusion = raw[split_idx:].strip()

        for header in ["Inclusion Criteria:", "Inclusion criteria:", "Inclusion:"]:
            inclusion = inclusion.replace(header, "").strip()

        return inclusion, exclusion

    def save(self, trials: list[Trial], condition: str) -> str:
        filename = condition.lower().replace(" ", "_") + "_trials.json"
        filepath = os.path.join(cfg.data_dir, "raw", filename)

        with open(filepath, "w") as f:
            json.dump([t.__dict__ for t in trials], f, indent=2)

        print(f"💾 Saved {len(trials)} trials → {filepath}")
        return filepath

    def preview(self, trials: list[Trial], n: int = 2):
        print(f"\n{'='*50}")
        print(f"  PREVIEW — first {n} of {len(trials)} trials")
        print(f"{'='*50}")

        for i, t in enumerate(trials[:n]):
            print(f"\n[{i+1}] {t.nct_id}")
            print(f"  Title   : {t.title[:65]}...")
            print(f"  Phase   : {t.phase}")
            print(f"  Status  : {t.status}")
            print(f"  Age     : {t.min_age} – {t.max_age}")
            print(f"  Sex     : {t.sex}")
            print(f"  Sponsor : {t.sponsor[:50]}")
            print(f"\n  INCLUSION (first 200 chars):")
            print(f"  {t.inclusion_criteria[:200]}...")
            print(f"\n  EXCLUSION (first 200 chars):")
            print(f"  {t.exclusion_criteria[:200]}...")
            print(f"  {'-'*48}")


if __name__ == "__main__":
    fetcher = ClinicalTrialsFetcher()

    conditions = ["type 2 diabetes", "hypertension", "breast cancer"]
    summary    = {}

    for condition in conditions:
        trials = fetcher.fetch(condition, n=50)
        if trials:
            fetcher.preview(trials, n=2)
            fetcher.save(trials, condition)
            summary[condition] = len(trials)
        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"  FETCH COMPLETE")
    print(f"{'='*50}")
    for cond, count in summary.items():
        print(f"  {cond:<25} → {count} trials")
    print(f"  {'TOTAL':<25} → {sum(summary.values())} trials")
    print(f"{'='*50}\n")