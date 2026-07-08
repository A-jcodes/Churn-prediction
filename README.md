# Telco Customer Churn: End-to-End ML System

Predicting customer churn for a telecom provider and serving predictions through a production API, from raw CSV to a containerized service deployable on AWS.

**Stack:** Python 3.11 · pandas · scikit-learn · MLflow · pytest · FastAPI · Docker · AWS (ECR + App Runner)

## The business problem

A telecom provider loses about 26.5% of its customers. Acquiring a new customer costs several times more than retaining an existing one, so the retention team needs to know which customers are likely to leave, and why, early enough to intervene with a targeted offer. The deliverable is not a notebook: it is a scored probability per customer, available on demand to the systems the retention team already uses. That requirement drives every design decision in this repo.

Data: the IBM Telco sample dataset via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). 7,043 customers, 21 columns covering demographics, subscribed services, contract and billing details, and the churn label.

## Problems found and how they were solved

**1. TotalCharges arrives as text.** Eleven rows contain blank strings, which silently turns the whole column into text. Investigation showed all eleven belong to customers with tenure = 0: brand new accounts that have not been billed yet. These are not missing values, so the fix is to impute 0, not the median, and not to drop the rows. New customers are exactly the population a churn model must score at serving time. The decision is implemented in `src/data.py` and enforced by a test that fails if the upstream data ever changes shape.

**2. Inconsistent encoding.** `SeniorCitizen` is 0/1 while every other binary column is Yes/No. Standardized to Yes/No so one encoder handles all categoricals and plots read consistently.

**3. Class imbalance.** At 26.5% churners, a model that predicts "no churn" for everyone scores 73% accuracy while being useless. The split is stratified, the linear model uses class weighting, and evaluation leads with ROC-AUC, PR-AUC, and recall on churners rather than accuracy.

**4. Train/serve skew risk.** All preprocessing (scaling, one-hot encoding) lives inside a single sklearn `Pipeline` that is persisted whole. The API deserializes that pipeline and feeds it raw-shaped records, so the exact transformations used in training run at serving time. There is no duplicated preprocessing code to drift out of sync.

## Methodology

Exploration and cleaning decisions are documented in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Headline findings: month-to-month contracts churn at roughly ten times the rate of two-year contracts, churn is heavily front-loaded in the first year of tenure, fiber optic customers churn far more than DSL despite being the premium tier, and electronic check payers churn at about double the rate of autopay customers.

Two candidates were trained on an 80/20 stratified split (`src/train.py`), with runs tracked in MLflow:

| Model | ROC-AUC | PR-AUC | Recall (churners) | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic regression (class weighted) | **0.842** | 0.633 | **0.783** | 0.614 | 0.738 |
| Hist gradient boosting | 0.834 | 0.642 | 0.516 | 0.571 | 0.794 |

The class-weighted logistic regression wins on the metrics that matter for this problem. The boosted model posts higher accuracy, which illustrates exactly why accuracy is the wrong headline: it buys that accuracy by missing nearly half of the churners, and a missed churner costs far more than a wasted retention offer. The simpler model also ships with interpretable coefficients the retention team can act on.

## Serving

`src/api.py` exposes the model through FastAPI with strict pydantic validation: every categorical field is a `Literal` type, so malformed requests fail with a clear 422 before reaching the model.

```
GET  /health    -> {"status": "ok", "model_loaded": true}
POST /predict   -> {"churn_probability": 0.8951, "churn_predicted": true, "risk_band": "high"}
```

The risk band (low / medium / high) gives the retention team a ready-made triage signal instead of a raw probability.

## Project structure

```
telco-churn/
├── app.py                     # Streamlit demo UI (local model or live API mode)
├── notebooks/01_eda.ipynb     # narrative EDA and cleaning decisions
├── src/
│   ├── data.py                # load + clean (single source of truth)
│   ├── train.py               # pipelines, evaluation, MLflow, model export
│   └── api.py                 # FastAPI serving layer
├── tests/test_data.py         # data quality contract
├── models/                    # trained pipeline (build artifact, gitignored)
├── data/                      # raw CSV (gitignored, provenance in data/README.md)
├── Dockerfile                 # slim, non-root, healthchecked serving image
├── requirements.txt           # runtime deps (what the Docker image installs)
├── requirements-dev.txt       # notebook, testing, and tracking deps
└── docs/DEPLOYMENT_AWS.md     # step-by-step AWS deployment
```

## Run it

```bash
# environment (uv)
uv venv --python 3.11 && source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
uv pip install -r requirements-dev.txt

# tests, training, API
python -m pytest tests/ -v
python -m src.train                                   # add --no-mlflow to skip tracking
uvicorn src.api:app --reload                          # http://localhost:8000/docs

# container
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api

# interactive demo UI
streamlit run app.py                                  # sidebar toggles local model vs live API
```

## Deployment

Two live surfaces, deployed separately:

- **Demo UI** on Streamlit Community Cloud: push the repo to GitHub, connect it at share.streamlit.io, point it at `app.py`. The small trained model (8 KB) is committed so the app loads it directly; switch the sidebar to Remote API mode to route predictions through the deployed service instead.
- **Production API** on AWS: image pushed to ECR, run on App Runner (with an ECS Fargate path for larger scale). Full walkthrough with commands, IAM notes, and cost estimates in [`docs/DEPLOYMENT_AWS.md`](docs/DEPLOYMENT_AWS.md).

## What I would do next

Threshold tuning against an explicit cost matrix (retention offer cost vs customer lifetime value), SHAP-based per-prediction explanations returned by the API, a great-expectations checkpoint gating retraining, and a GitHub Actions workflow that runs tests, rebuilds the image, and redeploys on merge.

---

*Austin Amadi · [github.com/A-jcodes](https://github.com/A-jcodes)*
