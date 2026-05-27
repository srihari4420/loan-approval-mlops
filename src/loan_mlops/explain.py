"""SHAP-based explanations for model decisions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _transform(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pre = pipeline.named_steps["preprocessor"]
    transformed = pre.transform(X)
    feature_names = list(pre.get_feature_names_out())
    return transformed, feature_names


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    sample_size: int | None = 1000,
) -> tuple[shap.Explanation, list[str]]:
    if sample_size and len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
        logger.info("Sampled inputs for SHAP", extra={"sample_size": sample_size})

    transformed, feature_names = _transform(pipeline, X)
    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "get_booster") or clf.__class__.__name__ in {"XGBClassifier", "LGBMClassifier"}:
        explainer = shap.TreeExplainer(clf)
        values = explainer(transformed)
    else:
        bg = shap.sample(transformed, 100, random_state=42)
        explainer = shap.KernelExplainer(clf.predict_proba, bg)
        raw = explainer.shap_values(transformed)
        values = shap.Explanation(
            values=raw[1] if isinstance(raw, list) else raw,
            base_values=np.full(len(transformed), explainer.expected_value),
            data=transformed,
            feature_names=feature_names,
        )

    return values, feature_names


def explain_single(
    pipeline: Pipeline,
    application: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    if len(application) != 1:
        raise ValueError(f"explain_single expects exactly 1 row, got {len(application)}")

    proba = float(pipeline.predict_proba(application)[0, 1])
    values, feature_names = compute_shap_values(pipeline, application, sample_size=None)

    shap_row = values.values[0]
    contributions = sorted(
        zip(feature_names, shap_row, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )

    risk_factors = [
        {"feature": name, "shap": float(v)} for name, v in contributions[:top_k] if v > 0
    ]
    protective_factors = [
        {"feature": name, "shap": float(v)} for name, v in contributions[-top_k:] if v < 0
    ]

    return {
        "default_probability": proba,
        "base_rate": float(values.base_values[0]),
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
    }


def save_global_summary(
    values: shap.Explanation,
    output_path: Path,
    max_display: int = 20,
) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot", extra={"path": str(output_path)})
    return output_path
