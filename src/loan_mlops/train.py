"""End-to-end training entrypoint, driven by Hydra config.

Usage:
    uv run python -m loan_mlops.train
    uv run python -m loan_mlops.train model.params.C=0.5
    uv run python -m loan_mlops.train data.missing_threshold=0.5
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from loan_mlops.data import load_clean
from loan_mlops.features import split_xy
from loan_mlops.logging_setup import set_correlation_id, setup_logging
from loan_mlops.model import build_baseline_pipeline, cross_validate, evaluate, save_model


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(level="INFO", json_format=False)
    cid = set_correlation_id()
    logger = logging.getLogger("train")

    logger.info("Run started", extra={"correlation_id": cid})
    logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    # Resolve paths relative to project root
    project_root = Path(hydra.utils.get_original_cwd())
    raw_path = project_root / cfg.data.raw_dir / cfg.data.train_file

    # Load and clean
    df = load_clean(
        raw_path=raw_path,
        sentinel_value=cfg.data.sentinel_days_employed,
        missing_threshold=cfg.data.missing_threshold,
    )
    X, y = split_xy(df, target_col=cfg.data.target_col, id_col=cfg.data.id_col)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )
    logger.info(
        "Train/test split",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )

    # Build and validate via CV
    pipeline = build_baseline_pipeline(
        X_train,
        model_params=OmegaConf.to_container(cfg.model.params),  # type: ignore[arg-type]
        random_state=cfg.random_state,
    )
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv_folds=cfg.training.cv_folds,
        scoring=cfg.training.scoring,
        random_state=cfg.random_state,
        n_jobs=cfg.training.n_jobs,
    )

    # Fit + evaluate on holdout
    logger.info("Fitting on full training set")
    pipeline.fit(X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test)

    # Save
    models_dir = project_root / "models"
    save_model(pipeline, output_dir=models_dir, name=f"{cfg.model.type}_v1")

    # Summary
    logger.info(
        "RUN COMPLETE",
        extra={
            "cv_auc_mean": cv_results["cv_score_mean"],
            "cv_auc_std": cv_results["cv_score_std"],
            "test_auc": test_metrics["test_auc"],
        },
    )


if __name__ == "__main__":
    main()
