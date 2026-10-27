"""Request bodies. Responses are plain JSON documented in each route's docstring."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _check_source(value: str) -> str:
    if value in ("reference", "annotation") or (value.startswith("pred:") and len(value) > 5):
        return value
    raise ValueError("source must be 'reference', 'annotation' or 'pred:<model_id>'")


class SegmentationRequest(BaseModel):
    model_id: str = Field("demo-classical", pattern=r"^[a-z0-9-]{1,40}$")
    mc_samples: int = Field(8, ge=0, le=32, description="MC-dropout passes (deep model only)")


class ComparisonRequest(BaseModel):
    model_id: str = Field(..., pattern=r"^[a-z0-9-]{1,40}$")
    reference: Literal["reference", "annotation"] = "reference"


class SourceRequest(BaseModel):
    source: str = Field("reference", description="'reference', 'annotation' or 'pred:<model_id>'")

    @field_validator("source")
    @classmethod
    def _valid_source(cls, value: str) -> str:
        return _check_source(value)


class DoseRequest(BaseModel):
    scenario_id: str = Field(..., pattern=r"^S[0-9]{1,2}$")
    target_source: str = "reference"

    @field_validator("target_source")
    @classmethod
    def _valid_source(cls, value: str) -> str:
        return _check_source(value)
