from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    """Incoming loan application. Mirrors the most-used columns from application_train.csv.

    Not every Home Credit column is exposed here — most are derived/internal scores
    that a real applicant wouldn't be filling in on a form. The model handles missing
    columns gracefully via the imputer in the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity (optional — for tracing)
    application_id: str | None = None

    # Demographics
    code_gender: Literal["M", "F", "XNA"] = Field(alias="CODE_GENDER")
    days_birth: int = Field(alias="DAYS_BIRTH", le=0, description="Days before today (negative)")
    cnt_children: int = Field(alias="CNT_CHILDREN", ge=0)

    # Employment & income
    name_income_type: str = Field(alias="NAME_INCOME_TYPE")
    name_education_type: str = Field(alias="NAME_EDUCATION_TYPE")
    amt_income_total: float = Field(alias="AMT_INCOME_TOTAL", ge=0)
    days_employed: int | None = Field(default=None, alias="DAYS_EMPLOYED")
    occupation_type: str | None = Field(default=None, alias="OCCUPATION_TYPE")

    # The loan
    name_contract_type: Literal["Cash loans", "Revolving loans"] = Field(alias="NAME_CONTRACT_TYPE")
    amt_credit: float = Field(alias="AMT_CREDIT", gt=0)
    amt_annuity: float | None = Field(default=None, alias="AMT_ANNUITY", gt=0)
    amt_goods_price: float | None = Field(default=None, alias="AMT_GOODS_PRICE", gt=0)

    # External scores — the big-signal features from EDA
    ext_source_1: float | None = Field(default=None, alias="EXT_SOURCE_1", ge=0, le=1)
    ext_source_2: float | None = Field(default=None, alias="EXT_SOURCE_2", ge=0, le=1)
    ext_source_3: float | None = Field(default=None, alias="EXT_SOURCE_3", ge=0, le=1)


class Factor(BaseModel):
    feature: str
    shap: float


class PredictionResponse(BaseModel):
    application_id: str | None
    decision: Literal["approve", "decline"]
    default_probability: float = Field(ge=0, le=1)
    threshold: float
    model_version: str
    risk_factors: list[Factor] = []
    protective_factors: list[Factor] = []
    correlation_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None