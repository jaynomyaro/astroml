"""Feature selection API router for AstroML.

Provides REST endpoints for filter, wrapper, embedded, and hybrid
feature selection with evaluation metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.preprocessing.feature_selection.embedded import EmbeddedSelector
from astroml.preprocessing.feature_selection.filter import FilterSelector
from astroml.preprocessing.feature_selection.hybrid import HybridSelector
from astroml.preprocessing.feature_selection.wrapper import WrapperSelector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["feature-selection"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class FeatureMatrixRequest(BaseModel):
    """Request body for feature selection on a matrix."""

    data: list[list[float]] = Field(..., description="Feature matrix rows")
    target: list[float] = Field(..., description="Target values")
    feature_names: list[str] | None = Field(default=None)
    method: str = Field(
        default="mutual_info",
        description="Selection method or strategy name",
    )
    k: int | None = Field(default=None, description="Number of features to select")
    threshold: float = Field(default=0.0, description="Score threshold")

    model_config = ConfigDict(extra="forbid")


class FeatureSetRequest(BaseModel):
    """Request body for hybrid selection with multiple strategies."""

    data: list[list[float]] = Field(...)
    target: list[float] = Field(...)
    feature_names: list[str] | None = Field(default=None)
    strategy: str = Field(default="vote", description="vote|rank_aggregation|intersection|union")
    methods: list[str] = Field(
        default=["mutual_info", "correlation", "variance"],
        description="Filter methods to ensemble",
    )
    k: int | None = Field(default=None)
    min_votes: int = Field(default=2)

    model_config = ConfigDict(extra="forbid")


class PipelineRequest(BaseModel):
    """Request body for pipeline selection."""

    data: list[list[float]] = Field(...)
    target: list[float] = Field(...)
    feature_names: list[str] | None = Field(default=None)
    steps: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Pipeline steps: [{type: filter|wrapper|embedded, method: ..., k: ...}, ...]"
        ),
    )

    model_config = ConfigDict(extra="forbid")


class SelectionResponse(BaseModel):
    selection_id: str
    method: str
    num_features_selected: int
    num_features_total: int
    selected_indices: list[int]
    scores: list[float]
    feature_names: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class EvaluationResponse(BaseModel):
    selection_id: str
    train_score: float | None = None
    test_score: float | None = None
    cv_score: float | None = None
    cv_std: float | None = None
    run_time_ms: float | None = None

    model_config = ConfigDict(extra="forbid")


# In-memory store for selections (in production, use a database)
_selections: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Filter endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/filter",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run filter-based feature selection",
)
async def filter_selection(request: FeatureMatrixRequest) -> SelectionResponse:
    try:
        X = np.array(request.data, dtype=np.float64)
        y = np.array(request.target, dtype=np.float64)

        selector = FilterSelector(
            method=request.method,
            k=request.k,
            threshold=request.threshold,
        )
        selector.fit(X, y, request.feature_names)
        result = selector.get_selection_result()

        import uuid as _uuid

        sel_id = _uuid.uuid4().hex[:12]
        _selections[sel_id] = {
            "result": result,
            "selector": selector,
            "X_shape": X.shape,
        }

        return SelectionResponse(
            selection_id=sel_id,
            method=f"filter-{request.method}",
            num_features_selected=result.num_features_selected,
            num_features_total=result.num_features_total,
            selected_indices=result.selected_indices,
            scores=[round(s, 6) for s in result.scores],
            feature_names=result.feature_names,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.exception("Error in filter selection")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Embedded endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/embedded",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run embedded feature selection",
)
async def embedded_selection(request: FeatureMatrixRequest) -> SelectionResponse:
    try:
        X = np.array(request.data, dtype=np.float64)
        y = np.array(request.target, dtype=np.float64)

        selector = EmbeddedSelector(
            method=request.method if request.method in ("lasso", "tree", "elasticnet") else "tree",
            threshold=request.threshold,
            k=request.k,
        )
        selector.fit(X, y, request.feature_names)
        result = selector.get_selection_result()

        import uuid as _uuid

        sel_id = _uuid.uuid4().hex[:12]
        _selections[sel_id] = {"result": result, "selector": selector, "X_shape": X.shape}

        return SelectionResponse(
            selection_id=sel_id,
            method=f"embedded-{request.method}",
            num_features_selected=result.num_features_selected,
            num_features_total=result.num_features_total,
            selected_indices=result.selected_indices,
            scores=[round(s, 6) for s in result.scores],
            feature_names=result.feature_names,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.exception("Error in embedded selection")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Hybrid / ensemble endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/hybrid",
    response_model=SelectionResponse,
    status_code=201,
    summary="Run hybrid feature selection (ensemble of filters)",
)
async def hybrid_selection(request: FeatureSetRequest) -> SelectionResponse:
    try:
        X = np.array(request.data, dtype=np.float64)
        y = np.array(request.target, dtype=np.float64)

        selectors = [
            (method, FilterSelector(method=method, k=request.k))
            for method in request.methods
        ]

        hybrid = HybridSelector(
            selectors=selectors,
            strategy=request.strategy,
            min_votes=request.min_votes,
            k=request.k,
        )
        hybrid.fit(X, y, request.feature_names)
        result = hybrid.get_selection_result()

        import uuid as _uuid

        sel_id = _uuid.uuid4().hex[:12]
        _selections[sel_id] = {"result": result, "selector": hybrid, "X_shape": X.shape}

        return SelectionResponse(
            selection_id=sel_id,
            method=f"hybrid-{request.strategy}",
            num_features_selected=result.num_features_selected,
            num_features_total=result.num_features_total,
            selected_indices=result.selected_indices,
            scores=[round(s, 6) for s in result.scores],
            feature_names=result.feature_names,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.exception("Error in hybrid selection")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Evaluation endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/feature-selection/evaluate/{selection_id}",
    response_model=EvaluationResponse,
    summary="Evaluate a feature selection result",
)
async def evaluate_selection(
    selection_id: str,
    data: list[list[float]] | None = None,
    target: list[float] | None = None,
    cv: int = Query(default=5, ge=2, le=20),
) -> EvaluationResponse:
    try:
        entry = _selections.get(selection_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Selection not found")

        if data is None or target is None:
            return EvaluationResponse(
                selection_id=selection_id,
                train_score=None,
                test_score=None,
                cv_score=None,
                cv_std=None,
            )

        import time

        X = np.array(data, dtype=np.float64)
        y = np.array(target, dtype=np.float64)

        selector = entry["selector"]

        t0 = time.monotonic()
        X_selected = selector.transform(X)

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            cv_scores = cross_val_score(model, X_selected, y, cv=min(cv, len(y) // 2))
            cv_score_mean = float(np.mean(cv_scores))
            cv_score_std = float(np.std(cv_scores))

            # Train on full dataset for reference
            model.fit(X_selected, y)
            train_score = float(model.score(X_selected, y))
        except ImportError:
            cv_score_mean = None
            cv_score_std = None
            train_score = None

        runtime = (time.monotonic() - t0) * 1000.0

        return EvaluationResponse(
            selection_id=selection_id,
            train_score=round(train_score, 4) if train_score is not None else None,
            cv_score=round(cv_score_mean, 4) if cv_score_mean is not None else None,
            cv_std=round(cv_score_std, 4) if cv_score_std is not None else None,
            run_time_ms=round(runtime, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error evaluating selection")
        raise HTTPException(status_code=500, detail=str(e))