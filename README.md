# FairCredit AI
### Explainable & Fair Credit Risk Decision-Support Platform

A final-year CSE project combining credit-risk prediction, model comparison, explainability, fairness auditing, and bias mitigation into one working system — not a conceptual sketch, an actual FastAPI + Streamlit + MySQL application you can run.

---

> **Running this on another computer?** See **[DEPLOYMENT.md](DEPLOYMENT.md)** — one command (`start.bat` on Windows, `./start.sh` on Linux/macOS), LAN access, backup/restore and troubleshooting. Verify a running deployment with `verify.bat` / `./verify.sh`.

---

## 1. Abstract

Automated credit-risk models can replicate or amplify historical lending disparities if deployed without transparency into how they decide and whether their outcomes differ across groups. FairCredit AI builds real predictive models on two independent, real datasets, and instruments both with genuine explainability (exact coefficients / real SHAP values) and fairness auditing (Fairlearn, including an actual before/after bias-mitigation comparison) rather than treating those as an afterthought. The system is presented as a human-in-the-loop decision-support tool: it produces a risk assessment for a human reviewer, never a final lending decision.

## 2. Problem statement

A credit model that is only "accurate" is an incomplete answer for a lending system. This project asks: can a credit-risk system be built that is simultaneously predictive, interpretable, and honest about where its outcomes diverge across demographic groups — and can that be demonstrated end-to-end, not just claimed?

## 3. Objectives

1. Build reproducible, leakage-checked pipelines for two real, independent datasets.
2. Compare multiple model families rather than assuming one is best, and select transparently (not "highest accuracy wins").
3. Explain every individual prediction with a real method — exact coefficients where the model is linear, real SHAP values where it is a tree ensemble.
4. Audit fairness with actual protected attributes present in each dataset (never invented ones), and show a genuine before/after bias-mitigation comparison with its real trade-offs.
5. Serve everything through a documented API and a dashboard usable by a non-technical evaluator.
6. Never claim more than the evidence supports — no "unbiased," no "production-ready," no fabricated metrics.

## 4. Novelty

Most course-project credit models stop at "train one model, report accuracy." This system integrates, in one running application: two independently-modelled datasets, 3-way model comparison with a documented selection formula, calibration (raw vs. isotonic-calibrated, with reliability curves), real SHAP + exact-coefficient explainability, a Fairlearn fairness audit with sample-size-aware reporting, an actual ThresholdOptimizer mitigation pass with before/after metrics, a model-governance page, and a MySQL-backed prediction audit trail — deployed via Docker Compose.

## 5. Two datasets — used as two independent tracks, not merged

**Track B — Washington State HMDA 2016** (mortgage lending): predicts `denied` (application denied vs. loan originated), derived transparently from `action_taken_name` (see §10).

**Track A — UCI Default of Credit Card Clients** (added this session): predicts `default` (next-month credit-card default), a target that ships already-binary in the source data.

These describe different populations, different decisions, and share no compatible feature schema. **They are not concatenated or relabelled into a shared target** — doing so would be scientifically indefensible (a mortgage denial is not the same event as a card default). The dashboard always shows which track/model produced a given result.

## 6. Architecture

```
                    BROWSER
                     |
                     v
         STREAMLIT FRONTEND        (port 8501)
   Dashboard | [Track B: Credit Prediction, Batch Prediction,
   Prediction History, Explainability, Fairness Audit, Model
   Performance, Dataset Information] | System Status | About |
   [Track A: Credit Card Prediction, Model Comparison,
   Credit Card Fairness] | Model Governance
                     |
                     | REST API (HTTP/JSON)
                     v
          FASTAPI BACKEND           (port 8000)
   Track B: /predict /predict/batch /model/info /model/feature-importance
            /metrics /fairness /dataset/info /predictions /predictions/stats /train
   Track A: /credit-card/predict /credit-card/predict/batch /credit-card/model/info
            /credit-card/model/comparison /credit-card/model/feature-importance
            /credit-card/fairness /credit-card/fairness/mitigation
            /credit-card/dataset/info /credit-card/predictions /credit-card/train
                     |
      +--------------+--------------+
      |              |              |
      v              v              v
 ML MODELS      FAIRNESS       EXPLAINABILITY
 (per track,    (Fairlearn,    (exact coefficients /
  joblib)       incl. real     real SHAP TreeExplainer)
                mitigation)
      |
      v
 MYSQL 8   (host 3307 -> container 3306)
 prediction_log (Track B) + credit_card_prediction_log (Track A)
```

Streamlit has no business logic or direct model/database access — every page calls FastAPI over HTTP (`frontend/services/api_client.py`).

## 7. Dataset descriptions

**Track A — UCI Default of Credit Card Clients:** 30,000 rows, 25 columns. Real financial-behaviour features (credit limit, 6 months of repayment status, bill amounts, payment amounts) plus demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`). No missing values; 35 exact-duplicate rows removed. `EDUCATION`/`MARRIAGE` contain undocumented category codes outside the published schema — bucketed into "Other/Unspecified" rather than guessed at.

**Track B — Washington State HMDA 2016:** 466,566 rows of real mortgage-application records; 327,889 usable after the documented target-derivation filter (§10 of the original build, unchanged this session).

## 8. Dataset-target differences

Track A's target (`default`) already exists in the source data as a bank-reported outcome. Track B's target (`denied`) is derived by this project from `action_taken_name`, keeping only the two directly comparable outcomes and excluding six ambiguous ones (withdrawn, incomplete, purchased-by-another-institution, approved-but-not-accepted, and two preapproval-only statuses) — see `backend/ml/feature_config.py`. These are different kinds of target and are never treated as equivalent.

## 9. Methodology (both tracks)

sklearn `Pipeline` + `ColumnTransformer` throughout: numeric imputation (median) + scaling, categorical imputation (most-frequent) + one-hot encoding (`handle_unknown="ignore"`), fitted **only** on training data, persisted together with the model as one joblib artifact. Fixed `random_state=42` everywhere for reproducibility. Preprocessing is never duplicated between training and inference — the API loads the exact same fitted pipeline used at training time.

## 10. Algorithms

**Track B:** Logistic Regression (`class_weight="balanced"`) — chosen for exact explainability (§12).

**Track A:** Logistic Regression, Random Forest (300 trees, depth 8, balanced), and XGBoost (300 rounds, depth 4) — all three actually trained and compared on the same held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 68.7% | 37.9% | 65.0% | 47.9% | 0.731 | 0.503 | 0.2049 |
| Random Forest | 77.9% | 50.0% | 58.5% | 53.9% | 0.778 | 0.558 | 0.1769 |
| XGBoost | 82.0% | 67.6% | 36.0% | 47.0% | **0.782** | **0.558** | **0.1341** |

**Selected model: Logistic Regression** — despite **not** having the highest ROC-AUC. Selection uses a documented weighted formula (`backend/ml/credit_card/trainer.py`): `0.35×ROC-AUC + 0.25×F1 + 0.20×calibration + 0.20×explainability`, where the explainability weight (1.0 for Logistic Regression vs. 0.6 for the tree ensembles) is an explicit, stated judgment call reflecting that its predictions can be explained exactly rather than via SHAP's estimated attributions — not a measured quantity. Composite scores: LR 0.6117, XGBoost 0.6038, RF 0.5856. Changing the weights would change the winner; that trade-off is visible in `/credit-card/model/info`, not hidden.

## 11. Model evaluation

Full metrics table above (Track A) and in `/metrics` (Track B: accuracy 64.5%, F1 43.8%, ROC-AUC 0.719). Confusion matrices, PR-AUC, and Brier scores are computed for every model, not just the selected one. Calibration is evaluated separately (§ below) — a model is not called "fair" or "good" from accuracy alone anywhere in this project.

**Calibration (Track A):** each model's raw probabilities are compared against isotonic-regression-calibrated probabilities, fit on the validation fold only (never train, never test). Reliability curves (predicted probability vs. actual fraction positive) and Brier scores before/after are on the Model Comparison page and in `/credit-card/model/comparison`.

## 12. Explainability

**Track B:** exact logistic-regression coefficient decomposition — `logit = intercept + Σ(coefficient × feature value)` is the literal computation, not an approximation.

**Track A:** the same exact-coefficient method for Logistic Regression; **real SHAP `TreeExplainer`** output for Random Forest and XGBoost (`backend/ml/credit_card/explain.py`) — actual Shapley-value estimates against the actual fitted trees, both per-prediction and as a global mean-|SHAP| importance ranking. If SHAP fails for any reason, the API returns `"available": false` with an explicit "Explanation unavailable" message — it never falls back to a fabricated explanation.

## 13. Fairness

Both tracks use **only protected attributes that actually exist in their dataset**:

- **Track B:** sex, race, ethnicity (HMDA fields). Real disparate impact ratios: sex 1.21, race 1.16, ethnicity 1.17 (comparison group denied more often in every case).
- **Track A:** **sex only** (`SEX`) — no race/ethnicity column exists in this dataset, so none is audited or invented. `AGE` is additionally checked as a labelled proxy dimension (35-and-over vs. under-35, a stated methodological choice) since, unlike SEX, AGE remains a model feature.

Fairness metrics computed: selection/denial rate by group, TPR/FPR by group, demographic parity difference/ratio, disparate impact ratio, equal opportunity difference, equalized odds difference — via Fairlearn's `MetricFrame` and metric functions, with group sample sizes always shown alongside the numbers. **This project never claims a model "is fair" or "is unbiased."**

## 14. Mitigation

**Track A** implements a real Fairlearn **ThresholdOptimizer** (constraint = equalized odds) as a post-processing mitigation, fit on train and evaluated on the held-out test set, using `SEX` as the sensitive feature. Actual before/after result on the selected model:

| | Before | After |
|---|---|---|
| Accuracy | 68.7% | 81.2% |
| Recall | 65.0% | 39.1% |
| F1 | 0.479 | 0.479 |
| Disparate impact ratio | 0.880 | 0.889 |
| Demographic parity difference | 0.049 | 0.017 |

Fairness improved (DI ratio closer to 1.0, parity gap shrank) while recall dropped substantially — a real, disclosed trade-off, not a "problem solved" claim. `GET /credit-card/fairness/mitigation` and the Credit Card Fairness page show both sides together, always.

## 15. Dashboard

Streamlit, 14 pages total. **Track B (HMDA):** Credit Prediction, Batch Prediction, Prediction History, Explainability, Fairness Audit, Model Performance, Dataset Information. **Track A (credit card):** Credit Card Prediction, Model Comparison, Credit Card Fairness. **Shared:** Dashboard (live aggregate stats), System Status (live-checked, not static), Model Governance (model card for both tracks, pulled live from the APIs), About.

## 16. API

FastAPI, `/docs` for interactive schemas. Track B and Track A endpoints listed in §6. Every input is Pydantic-validated against real category values (422 on invalid input, never a stack trace). A global exception handler returns structured 500s. CORS restricted to the configured frontend origin. The model is loaded once per track at process startup — never retrained per request; `/train` and `/credit-card/train` retrain explicitly and reload in-process.

## 17. Database

MySQL 8, SQLAlchemy + PyMySQL. Two audit-trail tables, created automatically at startup: `prediction_log` (Track B) and `credit_card_prediction_log` (Track A) — separate tables because the two tracks have different schemas and populations. Every successful prediction is logged best-effort (a database outage degrades to a logged warning, never a failed prediction). **Inside Docker, the backend connects to `mysql:3306`** (the Compose service name/port) — never `localhost:3307`, which only resolves from the Windows host. Host-accessible mapping stays `3307:3306` so it never collides with a MySQL already running on the host's port 3306.

## 18. Docker

`docker-compose.yml`: `mysql` (healthchecked via `mysqladmin ping`), `backend` (healthchecked via `/health`), `frontend` (healthchecked via Streamlit's own health endpoint), correct `depends_on: condition: service_healthy` ordering. No machine-specific paths anywhere in application code — everything is relative paths or environment variables.

## 19. Installation

```powershell
cd "<project-root>"
copy .env.example .env
docker compose build
docker compose up -d
docker compose ps
```
Place the raw datasets (not committed to git — see `.gitignore`):
```
data\raw\Washington_State_HDMA-2016.csv
data\raw\default_of_credit_card_clients.csv
```

## 20. Training

```
docker compose exec backend python scripts/prepare_data.py
docker compose exec backend python scripts/train_model.py
docker compose exec backend python scripts/evaluate_fairness.py

docker compose exec backend python scripts/prepare_credit_card_data.py
docker compose exec backend python scripts/train_credit_card_models.py
```
`data/` and `models/` are Docker-volume-mounted from the host, so these are visible to the running containers immediately.

## 21. Prediction

Open http://localhost:8501 → **Credit Prediction** (Track B) or **Credit Card Prediction** (Track A). Both return a risk assessment, probability, explanation, and a fixed reminder that this is not a final lending decision. Every prediction is written to its track's MySQL audit table automatically.

## 22. Testing

```
pip install -r requirements-dev.txt
pytest -q
```
**47 tests**, covering both tracks: data-pipeline logic, model loading, single/batch prediction, invalid-input 422s, fairness-metric math (including a synthetic-data correctness check), mitigation before/after, model-comparison/selection-formula sanity checks, database audit-trail read/write, and a regression test guarding against a real bug found this session (retraining one track used to silently wipe the other track's model registry entry — now fixed and permanently tested).

## 23. Project verification

```
python scripts/verify_project.py
```
36 checks (directories, files, both datasets, both tracks' processed splits and model artifacts, fairness reports, ROC-AUC sanity threshold, full pytest run) — PASS/FAIL output, no silent partial success.

## 24. Limitations

- Both datasets are single-snapshot, single-population historical data; results do not generalize to other markets, institutions, or time periods.
- Track A's fairness audit covers only SEX as a true protected attribute — no race/ethnicity data exists in this dataset.
- Track B's `class_weight="balanced"` shifts the predicted denial rate well above the true base rate — disclosed on the Model Performance page.
- ThresholdOptimizer mitigation (Track A) is evaluated once at a fixed operating point; it is a real, working demonstration of the mitigation/trade-off concept, not a production-grade fairness-constrained training pipeline.
- Both audit-trail tables are flat logs with no per-user access control — appropriate for a single-demo deployment, not a multi-tenant system.
- Neither track's prediction should be used for a real lending or credit decision without qualified human/institutional review.

## 25. Responsible-use statement

This is a demonstration and educational platform. It does not make lending decisions, does not guarantee approval or denial of any application, and does not claim to be free of bias. Every prediction page states this explicitly. Fairness results are statistical outcome comparisons on specific historical datasets — not a legal or regulatory determination of discrimination.

## 26. Future work

Bias-mitigation at training time (e.g. Fairlearn's `ExponentiatedGradient`) rather than only post-hoc thresholding; a counterfactual/what-if interactive panel; an Executive Dashboard aggregating both tracks' KPIs in one view; probability calibration for Track B; per-user authentication on the training/audit endpoints if deployed beyond a demo.
