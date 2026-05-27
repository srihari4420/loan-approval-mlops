# Loan Default Scoring — MLOps Platform

Production-shaped ML system for loan default prediction. End-to-end: data → model → API → audit log → cohort comparison.

![CI](https://github.com/srihari4420/loan-approval-mlops/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)

## What this is

A loan default classifier served via a FastAPI HTTP service. Built to demonstrate MLOps patterns at the depth a UK bank's platform team would expect, not just notebook-level ML.

### Capabilities

- **Model**: XGBoost classifier on Home Credit Default Risk, ROC-AUC ~0.76
- **Explainability**: SHAP values per prediction — every approve/decline decision comes with the top factors driving it
- **Champion/challenger routing**: two models served simultaneously, traffic split deterministically by application ID
- **Audit log**: every prediction recorded with model version, cohort, score, decision, factors, latency, correlation ID
- **Observability**: structured JSON logging with correlation IDs propagated through every request

### Architecture