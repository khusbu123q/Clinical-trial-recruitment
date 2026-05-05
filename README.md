# 🏥 Agentic AI for Clinical Trial Patient Recruitment

> A multi-agent deep learning system that automates end-to-end clinical trial patient recruitment using NLP, ClinicalBERT, and LangGraph agent orchestration.

---

## 📌 The Problem

Clinical trials are how new drugs get approved before reaching patients. But **85% of clinical trials fail to meet enrollment targets** — not because of the science, but because finding eligible patients is painfully slow.

Today a human coordinator manually reads through thousands of patient records, cross-referencing dozens of complex inclusion and exclusion criteria written in free text. This takes **3 to 6 months per trial** and costs millions of dollars. Every delay means a potentially life-saving drug reaches patients later.

---

## 💡 The Solution

This system automates the entire recruitment pipeline:

1. **Fetches** real trial protocols from ClinicalTrials.gov
2. **Parses** complex free-text eligibility criteria into structured logic
3. **Generates** realistic synthetic patient records with clinical notes
4. **Extracts** structured medical facts from clinical text using NLP
5. **Matches** patients to trials using rule-based logic + ClinicalBERT
6. **Ranks** candidates by eligibility score x enrollment likelihood
7. **Generates** personalised compliant outreach messages via LLM
8. **Enforces** safety through a dedicated compliance agent with veto power

---

## 🧠 Deep Learning Models

### Model 1 — Eligibility Classifier (ClinicalBERT Cross-Encoder)
- **Base model:** emilyalsentzer/Bio_ClinicalBERT (pre-trained on MIMIC-III clinical notes)
- **Architecture:** BERT encoder + classification head (Linear → GELU → Dropout → Linear)
- **Input format:** [CLS] Patient: {summary} [SEP] Criterion: {text} [SEP]
- **Output:** P(criterion met) — binary probability per criterion
- **Training data:** n2c2 2018 Cohort Selection Challenge dataset
- **Why cross-encoder:** Full bidirectional attention between patient tokens and criterion tokens captures interactions that dual-encoders miss

### Model 2 — Dropout Risk Predictor
- **Architecture:** XGBoost classifier on tabular patient features
- **Features:** Age, comorbidity count, medication count, visit frequency, distance to site
- **Output:** P(patient completes trial)
- **Final score:** eligibility_score x (1 - dropout_risk)

### Model 3 — Outreach Generator
- **Base:** GPT-4o via API
- **Training:** DPO (Direct Preference Optimisation) on human-rated message pairs
- **Constraints:** 200 words max, 8th-grade reading level, non-coercive, voluntary participation clearly stated
- **Safety:** Constitutional self-critique before compliance review

---

## 🤖 Agent Architecture

### Screener Agent
Uses a ReAct (Reasoning + Acting) loop with three tools:
- get_patient_labs — retrieves specific lab values
- get_patient_medications — retrieves medication history
- get_patient_diagnoses — retrieves active diagnoses

### Communicator Agent
- Generates personalised plain-language outreach messages
- Performs constitutional self-critique before submission
- Checks for false promises and pressure language

### Compliance Agent (Veto Gate)
Hard checks:
- Patient not on do-not-contact list
- Trial has valid IRB approval
- Message passes content safety review

### Orchestrator (LangGraph StateGraph)
Entry → Screener → ELIGIBLE → Communicator → Compliance → END
Entry → Screener → INELIGIBLE → Compliance → END

Shared RecruitmentState carries patient data, screening results, outreach message, compliance status, and full audit log across all agents.

---

## 📊 Results

| Metric | Value |
|---|---|
| Real trials downloaded | 150 |
| Trial criteria parsed | 751 |
| Rule-based criteria | 240 (32%) |
| LLM-flagged criteria | 511 (68%) |
| Synthetic patients | 1000 |
| Patient x trial evaluations | 150 |
| Eligible patients NCT06710340 | 39 / 50 |
| NER extraction rate | 100% |

---

## 🗂️ Project Structure

src/pipeline/fetch_trials.py — ClinicalTrials.gov API fetcher
src/pipeline/criteria_parser.py — Free-text criteria to structured logic
src/pipeline/generate_patients.py — Synthetic patient generator
src/pipeline/ner_pipeline.py — Clinical NER extraction
src/models/eligibility_matcher.py — Rule-based patient-trial matcher
src/models/eligibility_classifier.py — ClinicalBERT fine-tuning
src/models/dropout_predictor.py — XGBoost dropout risk model
src/models/outreach_generator.py — LLM message generation
src/agents/screener_agent.py — ReAct eligibility checker
src/agents/communicator_agent.py — Personalised outreach writer
src/agents/compliance_agent.py — Safety veto gate
src/agents/orchestrator.py — LangGraph state machine
src/evaluation/clinical_metrics.py — AUROC sensitivity Precision@K
src/evaluation/fairness_metrics.py — Disaggregated evaluation
app/dashboard.py — Streamlit coordinator dashboard

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Clinical NLP | ClinicalBERT scispaCy medspaCy |
| Deep Learning | PyTorch HuggingFace Transformers PEFT |
| Agent Framework | LangGraph LangChain |
| LLM | GPT-4o |
| Tabular ML | XGBoost |
| Vector Memory | ChromaDB |
| Experiment Tracking | Weights and Biases |
| Dashboard | Streamlit |

---

## ⚙️ Setup

Step 1 — Clone the repository

git clone https://github.com/khusbu123q/Clinical-trial-recruitment.git
cd Clinical-trial-recruitment

Step 2 — Create conda environment

conda create -n clinical_trial python=3.11 -y
conda activate clinical_trial

Step 3 — Install dependencies

pip install -r requirements.txt

Step 4 — Install scispaCy biomedical model

pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz

Step 5 — Set up API keys

Copy .env.example to .env and add your keys

Step 6 — Run the full pipeline

python src/pipeline/fetch_trials.py
python src/pipeline/generate_patients.py
python src/pipeline/ner_pipeline.py
python src/pipeline/criteria_parser.py
python src/models/eligibility_matcher.py

---

## 📋 API Keys Required

| Key | Where to get it | Required for |
|---|---|---|
| OPENAI_API_KEY | platform.openai.com | Agent reasoning and outreach |
| HF_TOKEN | huggingface.co | ClinicalBERT download |
| WANDB_API_KEY | wandb.ai | Experiment tracking |

---

## 📚 Datasets

| Dataset | Source | Used for |
|---|---|---|
| Clinical trial protocols | ClinicalTrials.gov public API | Trial criteria |
| Synthetic patient EHR | Synthea generator | Patient matching |
| n2c2 2018 Cohort Selection | portal.dbmi.hms.harvard.edu | Classifier training |
| MIMIC-IV optional | physionet.org | NLP fine-tuning |

---

## 🔬 Novelty and Contributions

1. Multi-agent architecture — First application of LangGraph multi-agent orchestration to clinical trial recruitment enabling dynamic tool-augmented eligibility reasoning rather than static classification

2. Compliance-by-design — Dedicated compliance agent with veto authority over patient outreach addressing a critical safety gap in prior work

3. Hybrid criteria parsing — Rule-based plus LLM fallback pipeline that handles 100% of real-world criteria regardless of complexity

4. Fairness-first evaluation — Mandatory disaggregated evaluation by demographic group with automatic bias flagging directly addressing the documented diversity problem in clinical trial recruitment

---

## 👩‍💻 Author

Khusbu Agarwal
Deep Learning Project — Clinical AI
GitHub: github.com/khusbu123q
