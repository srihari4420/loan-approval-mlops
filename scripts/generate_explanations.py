"""Generate the SHAP plots and example explanations for the README.

Run after training:
    uv run python scripts/generate_explanations.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from loan_mlops.data import load_clean
from loan_mlops.explain import compute_shap_values, explain_single, save_global_summary
from loan_mlops.features import split_xy
from loan_mlops.logging_setup import setup_logging
from loan_mlops.model import load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "xgboost_v1"
OUTPUT_DIR = ROOT / "docs" / "explanations"


def main() -> None:
    setup_logging()
    log = logging.getLogger("explain")

    df = load_clean(
        raw_path=ROOT / "data" / "raw" / "application_train.csv",
        sentinel_value=365243,
        missing_threshold=0.6,
    )
    X, y = split_xy(df, target_col="TARGET", id_col="SK_ID_CURR")

    pipeline = load_model(model_dir=ROOT / "models", name=MODEL_NAME)

    log.info("Computing global SHAP values on a sample of 1000")
    values, _ = compute_shap_values(pipeline, X, sample_size=1000)

    save_global_summary(values, OUTPUT_DIR / "shap_summary.png")

    # Three example explanations: one likely-default, one likely-repay, one borderline
    proba = pipeline.predict_proba(X)[:, 1]
    indices = {
        "high_risk": int(proba.argmax()),
        "low_risk": int(proba.argmin()),
        "borderline": int((abs(proba - 0.5)).argmin()),
    }

    examples = {}
    for label, idx in indices.items():
        row = X.iloc[[idx]]
        explanation = explain_single(pipeline, row, top_k=5)
        explanation["actual_target"] = int(y.iloc[idx])
        examples[label] = explanation
        log.info(f"{label}: p={explanation['default_probability']:.3f}, actual={y.iloc[idx]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "example_explanations.json").open("w") as f:
        json.dump(examples, f, indent=2)

    log.info("Done")


if __name__ == "__main__":
    main()