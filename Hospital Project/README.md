# Hospital Readmissions: Who Drives Excess Readmissions, and What Explains It? (CMS public data)

> **Status:** code complete; fill in the *Results* section after running on real CMS data.

## Business question
Medicare penalizes hospitals with excess 30-day readmissions. Which hospitals and conditions drive the problem, how much of the
signal is real versus statistical noise, what hospital characteristics explain it, and where should a health system or payer
focus intervention?

## Data (all public, no sensitive data)
| File (save as) | Source | Notes |
|---|---|---|
| `data/raw/hrrp.csv` | CMS Provider Data Catalog: *Hospital Readmissions Reduction Program* | Excess readmission ratio per hospital x condition (AMI, HF, PN, COPD, CABG, hip/knee) |
| `data/raw/hospital_info.csv` | CMS Provider Data Catalog: *Hospital General Information* | Type, ownership, emergency services, overall star rating |
| `data/raw/inpatient.csv` (optional) | data.cms.gov: *Medicare Inpatient Hospitals - by Provider and Service* | Used for payment per discharge |

Join key: CMS Certification Number (`Facility ID`), zero-padded to 6 characters.

**Important limitation:** this is *hospital-level* data, not patient-level. Findings describe hospitals, not individual patients,
and should not be read as patient-level risk factors.

## Run it
```bash
pip install -r requirements.txt
python -m src.load        # CSVs -> SQLite (data/hospital.db)
python -m src.analysis    # SQL analysis, figures, outputs/findings.md
python -m src.model       # grouped-CV models, calibration, subgroup checks -> outputs/model_results.md
streamlit run app.py      # interactive dashboard
```
If `load` reports missing columns, CMS renamed something; the error prints the columns found. Adjust the names in `src/load.py`.

## Methods
1. **SQL layer** (`sql/`): builds hospital x condition and hospital-level tables, then answers six business questions.
2. **Noise analysis:** approximates which "excess" ratios are statistically distinguishable from 1 using SE ~ 1/sqrt(observed readmissions), shown as a funnel plot.
3. **Predictive model:** predicts ratio > 1 from structural features using logistic regression and gradient boosting against a prevalence baseline.
   - GroupKFold by hospital (no hospital in both train and test).
   - CMS overall star rating is evaluated as a *separate* variant because readmissions feed into it (circular).
   - Reports AUC, Brier score, decile calibration, and AUC by ownership type.

## Results (fill in from `outputs/`)
- Key findings: copy from `outputs/findings.md` after verifying numbers.
- Model: AUC for structural features vs. baseline: ___. Interpretation: ___.
- Figures: `outputs/figures/`.

## Limitations (keep these honest)
- Hospital-level, observational data; no causal claims.
- Ratios are risk-adjusted by CMS, but unmeasured differences (case mix, social factors) remain.
- Small hospitals have noisy ratios; the significance test is an approximation.
- Penalty-relevant measure definitions and performance periods change across program years.

## Recommendations section (write after seeing results)
Example structure: (1) target the N persistent-excess hospitals first, (2) don't penalize or intervene on small-volume noise,
(3) if hospital features predict poorly, say so: closing the gap needs patient-level data (claims/EHR) and social-determinant
information.

## Extension modules
- **NLP/LLM layer:** see the companion `clinical-extraction` project (extraction + evaluation of clinical notes).
- **Cloud:** load `hospital_measure` to BigQuery and rebuild the dashboard in Looker Studio.
