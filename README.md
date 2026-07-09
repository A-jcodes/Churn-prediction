# Telco Customer Churn: End-to-End ML System

Predicting customer churn for a telecom provider and serving predictions through a production API, from raw CSV to a containerized service deployable on AWS.

**Stack:** Python 3.14.3 · pandas · scikit-learn · MLflow · pytest · FastAPI · Docker · AWS (ECR + App Runner)

## The business problem

A telecom provider loses about 26.5% of its customers. Acquiring a new customer costs several times more than retaining an existing one, so the retention team needs to know which customers are likely to leave, and why, early enough to intervene with a targeted offer. The deliverable is not a notebook: it is a scored probability per customer, available on demand to the systems the retention team already uses. That requirement drives every design decision in this repo.
