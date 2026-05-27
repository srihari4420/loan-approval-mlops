from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sklearn.pipeline import Pipeline

from loan_mlops.api.dependencies import get_expected_columns, get_model
from loan_mlops.api.schemas import (
    Factor,
    HealthResponse,
    LoanApplication,
    PredictionResponse,
)
from loan_mlops.api.settings import Settings, get_settings
from loan_mlops.explain import explain_single
from loan_mlops.logging_setup import set_correlation_id, setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(level=settings.log_level, json_format=settings.log_json)

    app = FastAPI(
        title="Loan Default Scoring API",
        version="0.1.0",
        description="Score loan applications and explain decisions.",
    )

    @app.middleware("http")
    async def correlation_and_timing(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        cid = request.headers.get("x-correlation-id") or set_correlation_id()
        set_correlation_id(cid)
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["x-correlation-id"] = cid
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(s: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        try:
            model = get_model()
            return HealthResponse(
                status="ok",
                model_loaded=model is not None,
                model_version=s.model_name,
            )
        except Exception as e:
            traceback.print_exc()
            logger.error("health check failed", extra={"error": repr(e), "type": type(e).__name__})
            return HealthResponse(status="degraded", model_loaded=False, model_version=None)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        application: LoanApplication,
        model: Annotated[Pipeline, Depends(get_model)],
        s: Annotated[Settings, Depends(get_settings)],
        expected_cols: Annotated[tuple[str, ...], Depends(get_expected_columns)],
    ) -> PredictionResponse:
        submitted = application.model_dump(by_alias=True, exclude={"application_id"})

        # Pad any columns the model was trained on but the caller didn't provide.
        # NaN flows through the pipeline's imputer the same way as during training.
        row_dict = {col: submitted.get(col, np.nan) for col in expected_cols}
        row = pd.DataFrame([row_dict], columns=list(expected_cols))

        try:
            proba = float(model.predict_proba(row)[0, 1])
        except Exception as e:
            logger.exception("scoring failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail="scoring failed") from e

        decision: Literal["approve", "decline"] = (
            "decline" if proba >= s.decision_threshold else "approve"
        )

        risk_factors: list[Factor] = []
        protective_factors: list[Factor] = []
        if s.enable_explanations:
            try:
                explanation = explain_single(model, row, top_k=5)
                risk_factors = [Factor(**f) for f in explanation["risk_factors"]]
                protective_factors = [Factor(**f) for f in explanation["protective_factors"]]
            except Exception as e:
                logger.warning("explanation failed", extra={"error": str(e)})

        return PredictionResponse(
            application_id=application.application_id,
            decision=decision,
            default_probability=proba,
            threshold=s.decision_threshold,
            model_version=s.model_name,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            correlation_id=set_correlation_id(),
        )

    return app


app = create_app()
