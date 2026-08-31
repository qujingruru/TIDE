"""Cluster-aware and held-out validation for behavioral dialogue measures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class ClusteredAssociation:
    """A standardized association with cluster-robust uncertainty."""

    beta: float
    lower: float
    upper: float
    partial_r: float
    standard_error: float
    p: float
    n: int
    clusters: int


@dataclass(frozen=True)
class ClusteredIncrement:
    """Incremental variance with a cluster-robust joint F test."""

    delta_r2: float
    f: float
    df_num: float
    df_den: float
    p: float
    n: int
    clusters: int


@dataclass(frozen=True)
class ClusteredModeration:
    """A clustered test of whether a metric slope differs between two groups."""

    interaction_beta: float
    lower: float
    upper: float
    standard_error: float
    p: float
    n: int
    clusters: int


@dataclass(frozen=True)
class HeldOutEvaluation:
    """Predictions and performance from grouped nested cross-validation."""

    r2: float
    mae: float
    correlation: float
    spearman: float
    n: int
    groups: int
    predictions: NDArray[np.float64]


@dataclass(frozen=True)
class PairedPerformanceDifference:
    """A paired held-out R-squared difference with a group-bootstrap interval."""

    delta_r2: float
    lower: float
    upper: float


@dataclass(frozen=True)
class RatingReliability:
    """Single-rater absolute-agreement ICC with a cluster-bootstrap interval."""

    icc: float
    icc_average: float
    lower: float
    upper: float
    average_lower: float
    average_upper: float
    n: int
    clusters: int


def _icc_two_way_random(values: NDArray[np.float64]) -> tuple[float, float]:
    complete = np.asarray(values, dtype=float)
    complete = complete[np.isfinite(complete).all(axis=1)]
    if complete.ndim != 2 or complete.shape[0] < 3 or complete.shape[1] < 2:
        raise ValueError("ICC(2,1) requires at least 3 complete targets and 2 raters")
    targets, raters = complete.shape
    grand_mean = float(complete.mean())
    target_means = complete.mean(axis=1)
    rater_means = complete.mean(axis=0)
    target_mean_square = raters * float(np.sum((target_means - grand_mean) ** 2)) / (targets - 1)
    rater_mean_square = targets * float(np.sum((rater_means - grand_mean) ** 2)) / (raters - 1)
    residuals = complete - target_means[:, None] - rater_means[None, :] + grand_mean
    error_mean_square = float(np.sum(residuals**2)) / ((targets - 1) * (raters - 1))
    denominator = (
        target_mean_square
        + (raters - 1) * error_mean_square
        + raters * (rater_mean_square - error_mean_square) / targets
    )
    average_denominator = target_mean_square + (rater_mean_square - error_mean_square) / targets
    if denominator == 0 or average_denominator == 0:
        raise ValueError("ICC(2,1) is undefined for the supplied ratings")
    icc_single = (target_mean_square - error_mean_square) / denominator
    icc_average = (target_mean_square - error_mean_square) / average_denominator
    return icc_single, icc_average


def _icc_two_way_random_single(values: NDArray[np.float64]) -> float:
    """Return ICC(2,1), retained as a focused internal compatibility helper."""

    return _icc_two_way_random(values)[0]


def cluster_bootstrap_icc_2_1(
    frame: pd.DataFrame,
    rater_columns: list[str],
    *,
    cluster: str = "dialogue_id",
    replicates: int = 1_000,
    seed: int = 42,
) -> RatingReliability:
    """Estimate ICC(2,1) and resample complete dialogues for its 95% interval."""

    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    data = cast(
        pd.DataFrame,
        frame.loc[:, [cluster, *rater_columns]].dropna().reset_index(drop=True),
    )
    cluster_values = cast(pd.Series, data[cluster]).drop_duplicates().to_numpy()
    if len(cluster_values) < 2:
        raise ValueError("Cluster bootstrap requires at least two dialogues")
    values = cast(pd.DataFrame, data.loc[:, rater_columns]).to_numpy(dtype=float)
    estimate, average_estimate = _icc_two_way_random(values)
    indices = {
        value: np.flatnonzero(cast(pd.Series, data[cluster]).to_numpy() == value)
        for value in cluster_values
    }
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    average_draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = generator.choice(cluster_values, size=len(cluster_values), replace=True)
        sampled_indices = np.concatenate([indices[value] for value in sampled])
        draws[replicate], average_draws[replicate] = _icc_two_way_random(values[sampled_indices])
    lower, upper = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
    average_lower, average_upper = np.quantile(
        average_draws[np.isfinite(average_draws)],
        [0.025, 0.975],
    )
    return RatingReliability(
        icc=float(estimate),
        icc_average=float(average_estimate),
        lower=float(lower),
        upper=float(upper),
        average_lower=float(average_lower),
        average_upper=float(average_upper),
        n=len(data),
        clusters=len(cluster_values),
    )


def _clustered_nested_test(
    data: pd.DataFrame,
    outcome: str,
    controls: pd.DataFrame,
    added_predictors: list[str],
    cluster: str,
) -> ClusteredIncrement:
    sm, _patsy, _stats = _require_dependencies()
    predictor_frame = cast(pd.DataFrame, data.loc[:, added_predictors])
    predictors = predictor_frame.apply(_standardize).reset_index(drop=True)
    baseline = controls.reset_index(drop=True)
    full_design = pd.concat([baseline, predictors], axis=1)
    outcome_values = cast(pd.Series, data[outcome])
    cluster_values = cast(pd.Series, data[cluster])
    restricted = sm.OLS(outcome_values, baseline).fit()
    full = sm.OLS(outcome_values, full_design).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_values, "use_correction": True},
        use_t=True,
    )
    restriction = np.zeros((len(added_predictors), full_design.shape[1]), dtype=float)
    restriction[:, -len(added_predictors) :] = np.eye(len(added_predictors))
    test = full.f_test(restriction)
    return ClusteredIncrement(
        delta_r2=float(full.rsquared - restricted.rsquared),
        f=float(np.asarray(test.fvalue).item()),
        df_num=float(test.df_num),
        df_den=float(test.df_denom),
        p=float(test.pvalue),
        n=len(data),
        clusters=int(cluster_values.nunique()),
    )


def _require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import patsy
        import statsmodels.api as sm
        from scipy import stats
    except ImportError as error:
        raise RuntimeError(
            "Validation dependencies are not installed. Run "
            "`pip install 'tide-dialogue[analysis]'` or `uv sync --extra analysis`."
        ) from error
    return sm, patsy, stats


def _standardize(values: pd.Series) -> pd.Series:
    deviation = values.std(ddof=1)
    if not np.isfinite(deviation) or deviation == 0:
        raise ValueError("Cannot standardize a constant or non-finite variable")
    return (values - values.mean()) / deviation


def _spline_basis(values: pd.Series, degrees_of_freedom: int) -> pd.DataFrame:
    _sm, patsy, _stats = _require_dependencies()
    basis = cast(
        pd.DataFrame,
        patsy.dmatrix(
            "bs(length, df=df, degree=3, include_intercept=False)",
            {"length": values.to_numpy(dtype=float), "df": degrees_of_freedom},
            return_type="dataframe",
        ).reset_index(drop=True),
    )
    for column in basis.columns:
        if column != "Intercept":
            basis[column] = _standardize(cast(pd.Series, basis[column]))
    return basis


def _spline_control_design(
    data: pd.DataFrame,
    length: str,
    spline_df: int,
    controls: list[str],
) -> pd.DataFrame:
    design = _spline_basis(cast(pd.Series, data[length]), spline_df)
    variable_controls = [
        column
        for column in controls
        if column != length and cast(pd.Series, data[column]).nunique() > 1
    ]
    for column in variable_controls:
        values = cast(pd.Series, data[column]).reset_index(drop=True)
        if column in {"n_chars", "n_words", "speaker_turn_progress"}:
            control_df = 3 if column == "speaker_turn_progress" else spline_df
            basis = _spline_basis(values, control_df).drop(columns="Intercept")
            basis.columns = [f"{column}_spline_{index + 1}" for index in range(basis.shape[1])]
            design = pd.concat([design, basis], axis=1)
        else:
            design = pd.concat(
                [design, _standardize(values).rename(column)],
                axis=1,
            )
    return design


def clustered_spline_association(
    frame: pd.DataFrame,
    metric: str,
    outcome: str,
    *,
    length: str = "n_chars",
    cluster: str = "dialogue_id",
    spline_df: int = 4,
    controls: list[str] | None = None,
) -> ClusteredAssociation:
    """Estimate a standardized metric association beyond nonlinear length.

    The point estimate is the metric coefficient from an OLS model with a
    cubic B-spline for length. Standard errors and p values are clustered by
    dialogue, which treats turns as observations within a dependent process.
    """

    sm, _dmatrix, _stats = _require_dependencies()
    additional_controls = controls or []
    columns = [metric, outcome, length, cluster, *additional_controls]
    data = cast(
        pd.DataFrame,
        frame.loc[:, columns].dropna().reset_index(drop=True),
    )
    cluster_values = cast(pd.Series, data[cluster])
    if len(data) < 30 or cluster_values.nunique() < 2:
        raise ValueError("Clustered association requires at least 30 rows and 2 clusters")
    standardized_metric = _standardize(cast(pd.Series, data[metric])).rename(metric)
    standardized_outcome = _standardize(cast(pd.Series, data[outcome]))
    design = pd.concat(
        [
            standardized_metric.reset_index(drop=True),
            _spline_control_design(data, length, spline_df, additional_controls),
        ],
        axis=1,
    )
    model = sm.OLS(standardized_outcome, design).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_values, "use_correction": True},
        use_t=True,
    )
    lower, upper = model.conf_int().loc[metric]
    return ClusteredAssociation(
        beta=float(model.params[metric]),
        lower=float(lower),
        upper=float(upper),
        partial_r=_partial_correlation_with_spline(
            data,
            metric,
            outcome,
            length,
            spline_df,
            additional_controls,
        ),
        standard_error=float(model.bse[metric]),
        p=float(model.pvalues[metric]),
        n=len(data),
        clusters=int(cluster_values.nunique()),
    )


def clustered_incremental_test(
    frame: pd.DataFrame,
    outcome: str,
    added_predictors: list[str],
    *,
    length: str = "n_chars",
    cluster: str = "dialogue_id",
    spline_df: int = 4,
    controls: list[str] | None = None,
) -> ClusteredIncrement:
    """Test a predictor family beyond a nonlinear length baseline."""

    additional_controls = controls or []
    columns = [outcome, length, cluster, *additional_controls, *added_predictors]
    data = cast(
        pd.DataFrame,
        frame.loc[:, columns].dropna().reset_index(drop=True),
    )
    cluster_values = cast(pd.Series, data[cluster])
    if len(data) < 50 or cluster_values.nunique() < 2:
        raise ValueError("Clustered increment requires at least 50 rows and 2 clusters")
    baseline = _spline_control_design(data, length, spline_df, additional_controls)
    return _clustered_nested_test(
        data,
        outcome,
        baseline,
        added_predictors,
        cluster,
    )


def clustered_linear_incremental_test(
    frame: pd.DataFrame,
    outcome: str,
    controls: list[str],
    added_predictors: list[str],
    *,
    cluster: str = "dialogue_id",
) -> ClusteredIncrement:
    """Test added predictors with arbitrary linear controls and clustered errors."""

    columns = [outcome, cluster, *controls, *added_predictors]
    data = cast(
        pd.DataFrame,
        frame.loc[:, columns].dropna().reset_index(drop=True),
    )
    cluster_values = cast(pd.Series, data[cluster])
    if len(data) < 50 or cluster_values.nunique() < 2:
        raise ValueError("Clustered increment requires at least 50 rows and 2 clusters")
    sm, _patsy, _stats = _require_dependencies()
    variable_controls = [
        column for column in controls if cast(pd.Series, data[column]).nunique() > 1
    ]
    baseline = sm.add_constant(
        cast(pd.DataFrame, data.loc[:, variable_controls]),
        has_constant="add",
    )
    return _clustered_nested_test(
        data,
        outcome,
        cast(pd.DataFrame, baseline),
        added_predictors,
        cluster,
    )


def clustered_group_interaction(
    frame: pd.DataFrame,
    metric: str,
    outcome: str,
    group: str,
    *,
    length: str = "n_chars",
    cluster: str = "dialogue_id",
    spline_df: int = 4,
    controls: list[str] | None = None,
) -> ClusteredModeration:
    """Test a standardized metric-by-group interaction with clustered uncertainty."""

    sm, _patsy, _stats = _require_dependencies()
    additional_controls = controls or []
    columns = [metric, outcome, group, length, cluster, *additional_controls]
    data = cast(
        pd.DataFrame,
        frame.loc[:, columns].dropna().reset_index(drop=True),
    )
    group_values = cast(pd.Series, data[group])
    cluster_values = cast(pd.Series, data[cluster])
    if group_values.nunique() != 2:
        raise ValueError("Group interaction requires exactly two observed groups")
    if len(data) < 50 or cluster_values.nunique() < 2:
        raise ValueError("Group interaction requires at least 50 rows and 2 clusters")
    standardized_metric = _standardize(cast(pd.Series, data[metric])).rename(metric)
    standardized_outcome = _standardize(cast(pd.Series, data[outcome]))
    centered_group = (group_values - group_values.mean()).rename(group)
    interaction_name = f"{metric}_by_{group}"
    interaction = (standardized_metric * centered_group).rename(interaction_name)
    design = pd.concat(
        [
            standardized_metric.reset_index(drop=True),
            centered_group.reset_index(drop=True),
            interaction.reset_index(drop=True),
            _spline_control_design(data, length, spline_df, additional_controls),
        ],
        axis=1,
    )
    model = sm.OLS(standardized_outcome, design).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_values, "use_correction": True},
        use_t=True,
    )
    lower, upper = model.conf_int().loc[interaction_name]
    return ClusteredModeration(
        interaction_beta=float(model.params[interaction_name]),
        lower=float(lower),
        upper=float(upper),
        standard_error=float(model.bse[interaction_name]),
        p=float(model.pvalues[interaction_name]),
        n=len(data),
        clusters=int(cluster_values.nunique()),
    )


def _partial_correlation_with_spline(
    frame: pd.DataFrame,
    metric: str,
    outcome: str,
    length: str,
    spline_df: int,
    controls: list[str],
) -> float:
    sm, _dmatrix, stats = _require_dependencies()
    metric_values = cast(pd.Series, frame[metric]).reset_index(drop=True)
    outcome_values = cast(pd.Series, frame[outcome]).reset_index(drop=True)
    basis = _spline_control_design(frame, length, spline_df, controls)
    residual_metric = sm.OLS(metric_values, basis).fit().resid
    residual_outcome = sm.OLS(outcome_values, basis).fit().resid
    return float(stats.pearsonr(residual_metric, residual_outcome)[0])


def cluster_bootstrap_partial_correlation(
    frame: pd.DataFrame,
    metric: str,
    outcome: str,
    *,
    length: str = "n_chars",
    cluster: str = "dialogue_id",
    spline_df: int = 4,
    controls: list[str] | None = None,
    replicates: int = 1_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return spline-adjusted partial r and a dialogue-bootstrap 95% CI."""

    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    additional_controls = controls or []
    columns = [metric, outcome, length, cluster, *additional_controls]
    data = cast(
        pd.DataFrame,
        frame.loc[:, columns].dropna().reset_index(drop=True),
    )
    cluster_series = cast(pd.Series, data[cluster])
    cluster_values = cluster_series.drop_duplicates().to_numpy()
    if len(data) < 30 or len(cluster_values) < 2:
        raise ValueError("Cluster bootstrap requires at least 30 rows and 2 clusters")
    estimate = _partial_correlation_with_spline(
        data,
        metric,
        outcome,
        length,
        spline_df,
        additional_controls,
    )
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    grouped = {
        value: cast(pd.DataFrame, group) for value, group in data.groupby(cluster, sort=False)
    }
    for index in range(replicates):
        sampled = generator.choice(cluster_values, size=len(cluster_values), replace=True)
        bootstrap = cast(
            pd.DataFrame,
            pd.concat([grouped[value] for value in sampled], ignore_index=True),
        )
        draws[index] = _partial_correlation_with_spline(
            bootstrap,
            metric,
            outcome,
            length,
            spline_df,
            additional_controls,
        )
    lower, upper = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
    return estimate, float(lower), float(upper)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjust a named family of p values with the Benjamini-Hochberg method."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted.tolist()
    finite = values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    scaled = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.minimum(monotone, 1.0)
    adjusted[finite_indices] = restored
    return adjusted.tolist()


def held_out_dialogue_evaluation(
    frame: pd.DataFrame,
    outcome: str,
    features: list[str],
    *,
    group: str = "dialogue_id",
    outer_splits: int = 10,
    inner_splits: int = 5,
    repeats: int = 1,
    strata: str | None = None,
    seed: int = 42,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> HeldOutEvaluation:
    """Evaluate a fixed feature family on dialogues excluded from model fitting."""

    try:
        from scipy import stats
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError(
            "Held-out validation requires `pip install 'tide-dialogue[analysis]'`."
        ) from error

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    columns = list(dict.fromkeys([outcome, group, *features, *([strata] if strata else [])]))
    data = cast(
        pd.DataFrame,
        frame.loc[:, columns].dropna(subset=[outcome, group]).reset_index(drop=True),
    )
    group_series = cast(pd.Series, data[group])
    group_values = group_series.to_numpy()
    unique_groups = group_series.nunique()
    if unique_groups < outer_splits or unique_groups < inner_splits + 1:
        raise ValueError("Not enough dialogues for the requested grouped cross-validation")
    target = cast(pd.Series, data[outcome]).to_numpy(dtype=float)
    prediction_sum = np.zeros(len(data), dtype=float)
    prediction_count = np.zeros(len(data), dtype=int)
    stratum_values = cast(pd.Series, data[strata]).to_numpy() if strata else None
    if strata:
        group_strata = cast(pd.DataFrame, data.loc[:, [group, strata]].drop_duplicates())
        if bool(group_strata[group].duplicated().to_numpy().any()):
            raise ValueError("The stratification value must be constant within each dialogue")
    for repeat in range(repeats):
        if strata:
            assert stratum_values is not None
            outer = StratifiedGroupKFold(
                n_splits=outer_splits,
                shuffle=True,
                random_state=seed + repeat,
            )
            outer_folds = outer.split(data, stratum_values, group_values)
        else:
            outer = GroupKFold(
                n_splits=outer_splits,
                shuffle=True,
                random_state=seed + repeat,
            )
            outer_folds = outer.split(data, target, group_values)
        for fold, (train, test) in enumerate(outer_folds):
            pipeline = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge()),
                ]
            )
            if strata:
                assert stratum_values is not None
                inner = StratifiedGroupKFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=seed + repeat * outer_splits + fold,
                )
                inner_folds = list(
                    inner.split(
                        data.iloc[train],
                        stratum_values[train],
                        group_values[train],
                    )
                )
            else:
                inner = GroupKFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=seed + repeat * outer_splits + fold,
                )
                inner_folds = list(
                    inner.split(
                        data.iloc[train],
                        target[train],
                        group_values[train],
                    )
                )
            search = GridSearchCV(
                pipeline,
                {"ridge__alpha": list(alphas)},
                scoring="neg_mean_absolute_error",
                cv=inner_folds,
            )
            search.fit(data.iloc[train][features], target[train])
            prediction_sum[test] += search.predict(data.iloc[test][features])
            prediction_count[test] += 1
    if bool((prediction_count != repeats).any()):
        raise RuntimeError("Repeated grouped cross-validation did not predict every row")
    predictions = prediction_sum / prediction_count
    correlation = np.asarray(stats.pearsonr(target, predictions)[0], dtype=float).item()
    spearman = np.asarray(stats.spearmanr(target, predictions)[0], dtype=float).item()
    return HeldOutEvaluation(
        r2=float(r2_score(target, predictions)),
        mae=float(mean_absolute_error(target, predictions)),
        correlation=correlation,
        spearman=spearman,
        n=len(data),
        groups=int(unique_groups),
        predictions=predictions,
    )


def cross_condition_transport_evaluation(
    frame: pd.DataFrame,
    outcome: str,
    features: list[str],
    source_value: str,
    target_value: str,
    *,
    condition: str = "group",
    group: str = "dialogue_id",
    inner_splits: int = 5,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> HeldOutEvaluation:
    """Fit in one dialogue condition and evaluate in the other condition."""

    try:
        from scipy import stats
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import GridSearchCV, GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError(
            "Transport validation requires `pip install 'tide-dialogue[analysis]'`."
        ) from error

    if source_value == target_value:
        raise ValueError("Source and target conditions must differ")
    columns = [outcome, condition, group, *features]
    data = cast(
        pd.DataFrame,
        frame.loc[:, columns].dropna(subset=[outcome, condition, group]).reset_index(drop=True),
    )
    source = cast(pd.DataFrame, data.loc[data[condition].eq(source_value)].reset_index(drop=True))
    target_frame = cast(
        pd.DataFrame,
        data.loc[data[condition].eq(target_value)].reset_index(drop=True),
    )
    source_groups = cast(pd.Series, source[group])
    target_groups = cast(pd.Series, target_frame[group])
    if source_groups.nunique() < inner_splits:
        raise ValueError("Not enough source dialogues for grouped penalty selection")
    if target_groups.nunique() < 2 or len(target_frame) < 3:
        raise ValueError("Transport evaluation requires at least two target dialogues")
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": list(alphas)},
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=inner_splits),
    )
    source_target = cast(pd.Series, source[outcome]).to_numpy(dtype=float)
    search.fit(
        source.loc[:, features],
        source_target,
        groups=source_groups.to_numpy(),
    )
    target = cast(pd.Series, target_frame[outcome]).to_numpy(dtype=float)
    predictions = np.asarray(search.predict(target_frame.loc[:, features]), dtype=float)
    correlation = np.asarray(stats.pearsonr(target, predictions)[0], dtype=float).item()
    spearman = np.asarray(stats.spearmanr(target, predictions)[0], dtype=float).item()
    return HeldOutEvaluation(
        r2=float(r2_score(target, predictions)),
        mae=float(mean_absolute_error(target, predictions)),
        correlation=correlation,
        spearman=spearman,
        n=len(target_frame),
        groups=int(target_groups.nunique()),
        predictions=predictions,
    )


def paired_group_bootstrap_delta_r2(
    target: NDArray[np.float64],
    baseline_predictions: NDArray[np.float64],
    full_predictions: NDArray[np.float64],
    groups: NDArray[Any],
    *,
    replicates: int = 1_000,
    seed: int = 42,
) -> PairedPerformanceDifference:
    """Compare two held-out prediction vectors by resampling complete groups."""

    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    arrays = [
        np.asarray(target, dtype=float),
        np.asarray(baseline_predictions, dtype=float),
        np.asarray(full_predictions, dtype=float),
        np.asarray(groups),
    ]
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Targets, predictions, and groups must have equal length")
    if not np.isfinite(np.column_stack(arrays[:3])).all():
        raise ValueError("Targets and predictions must be finite")

    def r2(observed: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        if denominator == 0:
            raise ValueError("Cannot compute R-squared for a constant target")
        return 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator

    target_values, baseline_values, full_values, group_values = arrays
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError("Group bootstrap requires at least two groups")
    indices = {group: np.flatnonzero(group_values == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_groups = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([indices[group] for group in sampled_groups])
        draws[replicate] = r2(
            target_values[sampled_indices],
            full_values[sampled_indices],
        ) - r2(
            target_values[sampled_indices],
            baseline_values[sampled_indices],
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return PairedPerformanceDifference(
        delta_r2=r2(target_values, full_values) - r2(target_values, baseline_values),
        lower=float(lower),
        upper=float(upper),
    )


def _format_p(value: float) -> str:
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def extract_markdown_tables(report: str) -> dict[str, pd.DataFrame]:
    """Extract report tables into named data frames for machine-readable export."""

    lines = report.splitlines()
    section = "table"
    counts: dict[str, int] = {}
    tables: dict[str, pd.DataFrame] = {}

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("## "):
            section = re.sub(r"[^a-z0-9]+", "_", line[3:].lower()).strip("_")
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            divider = cells(lines[index + 1])
            if divider and all(re.fullmatch(r":?-{3,}:?", value) for value in divider):
                header = cells(line)
                rows: list[list[str]] = []
                index += 2
                while index < len(lines) and lines[index].startswith("|"):
                    rows.append(cells(lines[index]))
                    index += 1
                counts[section] = counts.get(section, 0) + 1
                suffix = "" if counts[section] == 1 else f"_{counts[section]}"
                tables[f"{section}{suffix}"] = pd.DataFrame(
                    rows,
                    columns=pd.Index(header),
                )
                continue
        index += 1
    return tables


def with_sequence_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a single normalized within-speaker process clock and first-turn flag."""

    required = ["dialogue_id", "speaker_id", "turn"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Sequence controls require columns: " + ", ".join(missing))
    controlled = frame.copy()
    grouped = controlled.groupby(["dialogue_id", "speaker_id"], sort=False)
    speaker_turn_index = grouped["turn"].rank(method="first").astype(int)
    speaker_turn_count = grouped["turn"].transform("size").astype(int)
    denominator = (speaker_turn_count - 1).replace(0, np.nan)
    progress = ((speaker_turn_index - 1) / denominator).fillna(0.0)
    controlled["speaker_turn_progress"] = progress
    controlled["speaker_turn_progress_squared"] = progress**2
    controlled["first_speaker_turn"] = speaker_turn_index.eq(1).astype(int)
    return controlled


def robust_validation_report(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int = 1_000,
    cv_repeats: int = 5,
) -> str:
    """Create the cluster-aware, nonlinear, and held-out BRM validation report."""

    from tide.analysis import RATINGS, build_metric_trajectories
    from tide.pipeline import METRIC_COLUMNS

    required_structure = ["dialogue_id", "group", "turn", "speaker_id", "n_chars"]
    missing_structure = [column for column in required_structure if column not in frame.columns]
    if missing_structure:
        raise ValueError(
            "Validation input is missing structural columns: " + ", ".join(missing_structure)
        )
    scored = with_sequence_controls(
        cast(
            pd.DataFrame,
            frame.sort_values(["dialogue_id", "turn"], kind="mergesort").reset_index(drop=True),
        )
    )
    scored["group_ai"] = (scored["group"] == "AI").astype(int)
    scored["log_chars"] = np.log1p(scored["n_chars"])
    scored["sqrt_chars"] = np.sqrt(scored["n_chars"])
    scored["squared_chars"] = scored["n_chars"] ** 2
    if "n_words" in scored.columns:
        scored["log_words"] = np.log1p(scored["n_words"])
        scored["sqrt_words"] = np.sqrt(scored["n_words"])
        scored["squared_words"] = scored["n_words"] ** 2
    word_length_controls = ["n_words"] if "n_words" in scored.columns else []
    sequence_controls = [
        "speaker_turn_progress",
        "first_speaker_turn",
    ]
    association_controls = ["group_ai", *sequence_controls, *word_length_controls]
    readouts = [metric for metric in METRIC_COLUMNS if metric in scored.columns]
    dialogue_count = cast(pd.Series, scored["dialogue_id"]).nunique()
    lines = [
        "# BRM-oriented validation report",
        "",
        (
            f"Sample: {dialogue_count} dialogues and "
            f"{len(scored)} scored turns. Dialogue is the clustering and hold-out unit."
        ),
        (
            "Chronology is adjusted with normalized within-speaker dialogue progress "
            "and a first-speaker-turn indicator."
        ),
    ]
    lines.extend(
        [
            "",
            "## Criterion distributions",
            "",
            "| Criterion | N | Missing | Observed range | Median | "
            "At observed maximum | Human-AI | Peer |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rating, label in RATINGS.items():
        values = cast(pd.Series, scored[rating]).dropna()
        maximum = float(values.max())
        maximum_share = float(values.eq(maximum).mean())
        group_shares = {
            group_value: float(group_frame[rating].dropna().eq(maximum).mean())
            for group_value, group_frame in scored.groupby("group", sort=False)
        }
        lines.append(
            f"| {label} | {len(values)} | {len(scored) - len(values)} | "
            f"{float(values.min()):.2f}--{maximum:.2f} | {float(values.median()):.2f} | "
            f"{maximum_share:.1%} | {group_shares.get('AI', float('nan')):.1%} | "
            f"{group_shares.get('HM', float('nan')):.1%} |"
        )
    rater_pairs: dict[str, list[str]] = {}
    creative_pairs = {
        "Fluency": ["CR01_R1", "CR01_R2"],
        "Flexibility": ["CR02_R1", "CR02_R2"],
        "Originality": ["CR03_R1", "CR03_R2"],
    }
    creative_columns = [column for pair in creative_pairs.values() for column in pair]
    if all(column in scored.columns for column in creative_columns):
        scored["CR_composite_R1"] = scored[["CR01_R1", "CR02_R1", "CR03_R1"]].sum(
            axis=1,
            min_count=3,
        )
        scored["CR_composite_R2"] = scored[["CR01_R2", "CR02_R2", "CR03_R2"]].sum(
            axis=1,
            min_count=3,
        )
        creative_pairs["Creative-thinking composite"] = [
            "CR_composite_R1",
            "CR_composite_R2",
        ]
        rater_pairs.update(creative_pairs)
    critical_pairs = {
        "Critical-thinking analysis": ["CT01_R1", "CT01_R2"],
        "Critical-thinking evaluation": ["CT02_R1", "CT02_R2"],
        "Critical-thinking reasoning": ["CT03_R1", "CT03_R2"],
    }
    critical_columns = [column for pair in critical_pairs.values() for column in pair]
    if all(column in scored.columns for column in critical_columns):
        scored["CT_composite_R1"] = scored[["CT01_R1", "CT02_R1", "CT03_R1"]].mean(
            axis=1,
        )
        scored["CT_composite_R2"] = scored[["CT01_R2", "CT02_R2", "CT03_R2"]].mean(
            axis=1,
        )
        critical_pairs["Critical-thinking composite"] = [
            "CT_composite_R1",
            "CT_composite_R2",
        ]
        rater_pairs.update(critical_pairs)
    if rater_pairs:
        lines.extend(
            [
                "",
                "## Human-rating criterion reliability",
                "",
                "| Criterion | ICC(2,1) | 95% CI | ICC(2,2) | 95% CI | N | Dialogues |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, columns in rater_pairs.items():
            reliability = cluster_bootstrap_icc_2_1(
                scored,
                columns,
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {label} | {reliability.icc:.3f} | "
                f"[{reliability.lower:.3f}, {reliability.upper:.3f}] | "
                f"{reliability.icc_average:.3f} | "
                f"[{reliability.average_lower:.3f}, {reliability.average_upper:.3f}] | "
                f"{reliability.n} | {reliability.clusters} |"
            )
    lines.extend(
        [
            "",
            "## Nonlinear length and interlocutor-type control with dialogue-clustered uncertainty",
            "",
            "| Metric | Outcome | Standardized beta | 95% CI | Partial r | "
            "Cluster-robust p | BH q | N | Dialogues |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    associations: list[tuple[str, str, ClusteredAssociation]] = []
    for metric in readouts:
        for rating, label in RATINGS.items():
            result = clustered_spline_association(
                scored,
                metric,
                rating,
                controls=association_controls,
            )
            associations.append((metric, label, result))
    adjusted = benjamini_hochberg([result.p for _metric, _label, result in associations])
    for (metric, label, result), q_value in zip(associations, adjusted, strict=True):
        lines.append(
            f"| {metric} | {label} | {result.beta:.3f} | "
            f"[{result.lower:.3f}, {result.upper:.3f}] | {result.partial_r:.3f} | "
            f"{_format_p(result.p)} | {_format_p(q_value)} | "
            f"{result.n} | {result.clusters} |"
        )

    moderation_rows: list[
        tuple[str, str, ClusteredAssociation, ClusteredAssociation, ClusteredModeration]
    ] = []
    ai_turns = cast(pd.DataFrame, scored.loc[scored["group_ai"].eq(1)])
    peer_turns = cast(pd.DataFrame, scored.loc[scored["group_ai"].eq(0)])
    for metric in readouts:
        for rating, label in RATINGS.items():
            ai_result = clustered_spline_association(
                ai_turns,
                metric,
                rating,
                controls=[*sequence_controls, *word_length_controls],
            )
            peer_result = clustered_spline_association(
                peer_turns,
                metric,
                rating,
                controls=[*sequence_controls, *word_length_controls],
            )
            moderation = clustered_group_interaction(
                scored,
                metric,
                rating,
                "group_ai",
                controls=[*sequence_controls, *word_length_controls],
            )
            moderation_rows.append((metric, label, ai_result, peer_result, moderation))
    moderation_q = benjamini_hochberg(
        [moderation.p for _metric, _label, _ai, _peer, moderation in moderation_rows]
    )
    lines.extend(
        [
            "",
            "## Interlocutor-type stability",
            "",
            "| Metric | Outcome | Human-AI partial r | Peer partial r | Interaction p | BH q |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for (metric, label, ai_result, peer_result, moderation), q_value in zip(
        moderation_rows,
        moderation_q,
        strict=True,
    ):
        lines.append(
            f"| {metric} | {label} | {ai_result.partial_r:.3f} | "
            f"{peer_result.partial_r:.3f} | {_format_p(moderation.p)} | "
            f"{_format_p(q_value)} |"
        )

    estimate, lower, upper = cluster_bootstrap_partial_correlation(
        scored,
        "lexical_entropy",
        "CR01_avg",
        controls=association_controls,
        replicates=bootstrap_replicates,
    )
    lines.extend(
        [
            "",
            (
                "The spline-adjusted lexical-entropy/Fluency partial correlation was "
                f"r = {estimate:.3f}, dialogue-bootstrap 95% CI "
                f"[{lower:.3f}, {upper:.3f}]."
            ),
        ]
    )
    if "n_words" in scored.columns:
        word_result = clustered_spline_association(
            scored,
            "lexical_entropy",
            "CR01_avg",
            length="n_words",
            controls=[
                "group_ai",
                *sequence_controls,
                "n_chars",
            ],
        )
        lines.append(
            "Using segmented-word length instead gave "
            f"partial r = {word_result.partial_r:.3f}, "
            f"p {_format_p(word_result.p)}."
        )

    spline_sensitivity = [
        clustered_spline_association(
            scored,
            "lexical_entropy",
            "CR01_avg",
            spline_df=degrees_of_freedom,
            controls=association_controls,
        ).partial_r
        for degrees_of_freedom in range(4, 8)
    ]
    lines.append(
        "Across cubic spline specifications with 4 to 7 degrees of freedom, "
        "the lexical-entropy/Fluency partial correlation ranged from "
        f"{min(spline_sensitivity):.3f} to {max(spline_sensitivity):.3f}."
    )

    families = {
        "Lexical entropy beyond MATTR": (
            ["lexical_entropy"],
            [*association_controls, "mattr"],
        ),
        "Contextual unexpectedness": (
            [
                column
                for column in [
                    "surprisal_mean",
                    "surprisal_sent_max",
                    "delta_surprisal",
                    "peak_break",
                ]
                if column in scored.columns
            ],
            association_controls,
        ),
        "Semantic relation": (
            [
                column
                for column in ["sem_dist_partner", "sem_dist_self", "self_novelty"]
                if column in scored.columns
            ],
            association_controls,
        ),
        "All TIDE readouts": (readouts, association_controls),
    }
    lines.extend(
        [
            "",
            "## Increment beyond a nonlinear length baseline",
            "",
            "| Predictor family | Outcome | Delta R2 | Cluster-robust F | p | BH q | N |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    increments: list[tuple[str, str, ClusteredIncrement]] = []
    for family, (predictors, controls) in families.items():
        for rating, label in RATINGS.items():
            result = clustered_incremental_test(
                scored,
                rating,
                predictors,
                controls=controls,
            )
            increments.append((family, label, result))
    adjusted_increments = benjamini_hochberg([result.p for _family, _label, result in increments])
    for (family, label, result), q_value in zip(
        increments,
        adjusted_increments,
        strict=True,
    ):
        lines.append(
            f"| {family} | {label} | {result.delta_r2:.3f} | {result.f:.2f} | "
            f"{_format_p(result.p)} | {_format_p(q_value)} | {result.n} |"
        )

    length_features = [
        "group_ai",
        *sequence_controls,
        "n_chars",
        "log_chars",
        "sqrt_chars",
        "squared_chars",
    ]
    if "n_words" in scored.columns:
        length_features.extend(["n_words", "log_words", "sqrt_words", "squared_words"])
    semantic_descriptors = [
        metric for metric in ["mattr", "sem_dist_partner", "sem_dist_self"] if metric in readouts
    ]
    information_readouts = [
        metric
        for metric in [
            "lexical_entropy",
            "surprisal_mean",
            "surprisal_sent_max",
            "delta_surprisal",
            "peak_break",
        ]
        if metric in readouts
    ]
    feature_sets = {
        "Interlocutor-type baseline": ["group_ai"],
        "Structural length baseline": length_features,
        "Established text descriptors": [*length_features, *semantic_descriptors],
        "Information readouts": [*length_features, *information_readouts],
        "Full TIDE family": [*length_features, *readouts],
    }
    lines.extend(
        [
            "",
            "## Held-out dialogue evaluation",
            "",
            f"{cv_repeats} repeated outer partitions were stratified by interlocutor type. "
            "Ridge penalties "
            "were selected inside each training fold. No turn from a held-out dialogue entered "
            "fitting or penalty selection.",
            "",
            "| Feature family | Outcome | Held-out R2 | MAE | "
            "Predicted-observed r | Spearman rho | N | Dialogues |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    evaluations: dict[tuple[str, str], HeldOutEvaluation] = {}
    for family, features in feature_sets.items():
        for rating, label in RATINGS.items():
            result = held_out_dialogue_evaluation(
                scored,
                rating,
                features,
                repeats=cv_repeats,
                strata="group",
            )
            evaluations[(family, rating)] = result
            lines.append(
                f"| {family} | {label} | {result.r2:.3f} | "
                f"{result.mae:.3f} | {result.correlation:.3f} | {result.spearman:.3f} | "
                f"{result.n} | {result.groups} |"
            )

    lines.extend(
        [
            "",
            "## Paired held-out improvement of the full TIDE family",
            "",
            "Intervals resample complete dialogues and preserve the paired "
            "out-of-fold predictions.",
            "",
            "| Outcome | Delta R2 over structural length | 95% CI | "
            "Delta R2 over established descriptors | 95% CI | N |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rating, label in RATINGS.items():
        data = cast(pd.DataFrame, scored.loc[:, [rating, "dialogue_id"]].dropna())
        length_difference = paired_group_bootstrap_delta_r2(
            cast(pd.Series, data[rating]).to_numpy(dtype=float),
            evaluations[("Structural length baseline", rating)].predictions,
            evaluations[("Full TIDE family", rating)].predictions,
            cast(pd.Series, data["dialogue_id"]).to_numpy(),
            replicates=bootstrap_replicates,
        )
        descriptor_difference = paired_group_bootstrap_delta_r2(
            cast(pd.Series, data[rating]).to_numpy(dtype=float),
            evaluations[("Established text descriptors", rating)].predictions,
            evaluations[("Full TIDE family", rating)].predictions,
            cast(pd.Series, data["dialogue_id"]).to_numpy(),
            replicates=bootstrap_replicates,
        )
        lines.append(
            f"| {label} | {length_difference.delta_r2:.3f} | "
            f"[{length_difference.lower:.3f}, {length_difference.upper:.3f}] | "
            f"{descriptor_difference.delta_r2:.3f} | "
            f"[{descriptor_difference.lower:.3f}, {descriptor_difference.upper:.3f}] | "
            f"{len(data)} |"
        )

    truncation_columns = {
        "sem_dist_partner": "sem_dist_partner_truncate",
        "sem_dist_self": "sem_dist_self_truncate",
        "self_novelty": "self_novelty_truncate",
    }
    if set(truncation_columns.values()).issubset(scored.columns):
        truncated_readouts = [truncation_columns.get(metric, metric) for metric in readouts]
        lines.extend(
            [
                "",
                "## Embedding long-text sensitivity",
                "",
                "The canonical token-weighted chunk aggregation is compared with retaining "
                "only the first 512-token embedding window.",
                "",
                "| Outcome | Canonical R2 | Truncation R2 | Truncation minus canonical R2 | "
                "95% CI | Canonical rho | Truncation rho |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rating, label in RATINGS.items():
            canonical = evaluations[("Full TIDE family", rating)]
            truncated = held_out_dialogue_evaluation(
                scored,
                rating,
                [*length_features, *truncated_readouts],
                repeats=cv_repeats,
                strata="group",
            )
            target_data = cast(
                pd.DataFrame,
                scored.loc[:, [rating, "dialogue_id"]].dropna(),
            )
            difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, target_data[rating]).to_numpy(dtype=float),
                canonical.predictions,
                truncated.predictions,
                cast(pd.Series, target_data["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {label} | {canonical.r2:.3f} | {truncated.r2:.3f} | "
                f"{difference.delta_r2:.3f} | "
                f"[{difference.lower:.3f}, {difference.upper:.3f}] | "
                f"{canonical.spearman:.3f} | {truncated.spearman:.3f} |"
            )

    if "source_quality_flag" in scored.columns and bool(scored["source_quality_flag"].any()):
        flag_counts = scored.groupby("dialogue_id")["source_quality_flag"].nunique()
        if bool((flag_counts > 1).any()):
            raise ValueError("source_quality_flag must be constant within each dialogue")
        quality_scored = cast(
            pd.DataFrame,
            scored.loc[~scored["source_quality_flag"].astype(bool)].copy(),
        )
        lines.extend(
            [
                "",
                "## Source-quality exclusion sensitivity",
                "",
                "Complete flagged dialogues are excluded; rated turns are not re-segmented.",
                "",
                "| Outcome | Full TIDE R2 | Delta R2 over structural length | 95% CI | "
                "Delta R2 over established descriptors | 95% CI | N | Dialogues |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rating, label in RATINGS.items():
            structural = held_out_dialogue_evaluation(
                quality_scored,
                rating,
                feature_sets["Structural length baseline"],
                repeats=cv_repeats,
                strata="group",
            )
            descriptor = held_out_dialogue_evaluation(
                quality_scored,
                rating,
                feature_sets["Established text descriptors"],
                repeats=cv_repeats,
                strata="group",
            )
            full = held_out_dialogue_evaluation(
                quality_scored,
                rating,
                feature_sets["Full TIDE family"],
                repeats=cv_repeats,
                strata="group",
            )
            target_data = cast(
                pd.DataFrame,
                quality_scored.loc[:, [rating, "dialogue_id"]].dropna(),
            )
            length_difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, target_data[rating]).to_numpy(dtype=float),
                structural.predictions,
                full.predictions,
                cast(pd.Series, target_data["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            descriptor_difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, target_data[rating]).to_numpy(dtype=float),
                descriptor.predictions,
                full.predictions,
                cast(pd.Series, target_data["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {label} | {full.r2:.3f} | {length_difference.delta_r2:.3f} | "
                f"[{length_difference.lower:.3f}, {length_difference.upper:.3f}] | "
                f"{descriptor_difference.delta_r2:.3f} | "
                f"[{descriptor_difference.lower:.3f}, {descriptor_difference.upper:.3f}] | "
                f"{len(target_data)} | {full.groups} |"
            )

    lines.extend(
        [
            "",
            "## Within-condition held-out evaluation",
            "",
            "Complete dialogues are held out within each interlocutor condition.",
            "",
            "| Condition | Outcome | Descriptor R2 | Full TIDE R2 | Delta R2 | 95% CI | "
            "Full Spearman rho | N | Dialogues |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_value, condition_label in [("AI", "Human-AI"), ("HM", "Peer")]:
        condition_scored = cast(
            pd.DataFrame,
            scored.loc[scored["group"].eq(condition_value)].copy(),
        )
        for rating, label in RATINGS.items():
            descriptor = held_out_dialogue_evaluation(
                condition_scored,
                rating,
                feature_sets["Established text descriptors"],
                repeats=cv_repeats,
            )
            full = held_out_dialogue_evaluation(
                condition_scored,
                rating,
                feature_sets["Full TIDE family"],
                repeats=cv_repeats,
            )
            target_data = cast(
                pd.DataFrame,
                condition_scored.loc[:, [rating, "dialogue_id"]].dropna(),
            )
            difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, target_data[rating]).to_numpy(dtype=float),
                descriptor.predictions,
                full.predictions,
                cast(pd.Series, target_data["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {condition_label} | {label} | {descriptor.r2:.3f} | {full.r2:.3f} | "
                f"{difference.delta_r2:.3f} | "
                f"[{difference.lower:.3f}, {difference.upper:.3f}] | "
                f"{full.spearman:.3f} | {full.n} | {full.groups} |"
            )

    reference_baseline_features = [*length_features, "mattr"]
    reference_families = {
        "Preceding partner turn": ["sem_dist_partner"],
        "Preceding self turn": ["sem_dist_self"],
        "Most similar prior self turn": ["self_novelty"],
    }
    lines.extend(
        [
            "",
            "## Held-out comparison of semantic reference frames",
            "",
            "Each reference model is compared with the same structural-plus-MATTR baseline.",
            "",
            "| Outcome | Reference frame | Held-out R2 | Delta R2 | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for rating, label in RATINGS.items():
        baseline = held_out_dialogue_evaluation(
            scored,
            rating,
            reference_baseline_features,
            repeats=cv_repeats,
            strata="group",
        )
        target_data = cast(pd.DataFrame, scored.loc[:, [rating, "dialogue_id"]].dropna())
        for reference, reference_readouts in reference_families.items():
            reference_model = held_out_dialogue_evaluation(
                scored,
                rating,
                [*reference_baseline_features, *reference_readouts],
                repeats=cv_repeats,
                strata="group",
            )
            difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, target_data[rating]).to_numpy(dtype=float),
                baseline.predictions,
                reference_model.predictions,
                cast(pd.Series, target_data["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {label} | {reference} | {reference_model.r2:.3f} | "
                f"{difference.delta_r2:.3f} | "
                f"[{difference.lower:.3f}, {difference.upper:.3f}] |"
            )

    ablation_blocks = {
        "Lexical distribution": ["lexical_entropy"],
        "Contextual unexpectedness": information_readouts[1:],
        "Semantic relation": [
            metric
            for metric in ["sem_dist_partner", "sem_dist_self", "self_novelty"]
            if metric in readouts
        ],
    }
    lines.extend(
        [
            "",
            "## Held-out metric-family ablation",
            "",
            "Positive values indicate performance lost when the named block is removed.",
            "",
            "| Outcome | Removed block | Full R2 | Ablated R2 | R2 loss | 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for rating, label in RATINGS.items():
        full = evaluations[("Full TIDE family", rating)]
        target_data = cast(pd.DataFrame, scored.loc[:, [rating, "dialogue_id"]].dropna())
        for block, removed in ablation_blocks.items():
            ablated_readouts = [metric for metric in readouts if metric not in removed]
            ablated = held_out_dialogue_evaluation(
                scored,
                rating,
                [*length_features, *ablated_readouts],
                repeats=cv_repeats,
                strata="group",
            )
            difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, target_data[rating]).to_numpy(dtype=float),
                ablated.predictions,
                full.predictions,
                cast(pd.Series, target_data["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {label} | {block} | {full.r2:.3f} | {ablated.r2:.3f} | "
                f"{difference.delta_r2:.3f} | "
                f"[{difference.lower:.3f}, {difference.upper:.3f}] |"
            )

    lines.extend(
        [
            "",
            "## Cross-condition transport",
            "",
            "Models are tuned only on source-condition dialogues and evaluated on all "
            "dialogues in the other condition. This is transport across interlocutor type "
            "within one debate topic, not evidence of cross-task generalization.",
            "",
            "| Direction | Outcome | Descriptor R2 | Full TIDE R2 | Delta R2 | "
            "95% CI | Descriptor MAE | Full TIDE MAE | Full predicted-observed r | "
            "Full Spearman rho |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, target_condition, direction in [
        ("HM", "AI", "Peer to human-AI"),
        ("AI", "HM", "Human-AI to peer"),
    ]:
        for rating, label in RATINGS.items():
            descriptor = cross_condition_transport_evaluation(
                scored,
                rating,
                feature_sets["Established text descriptors"],
                source,
                target_condition,
            )
            full = cross_condition_transport_evaluation(
                scored,
                rating,
                feature_sets["Full TIDE family"],
                source,
                target_condition,
            )
            target_data = cast(
                pd.DataFrame,
                scored.loc[
                    scored["group"].eq(target_condition),
                    [rating, "dialogue_id"],
                ].dropna(),
            )
            difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, target_data[rating]).to_numpy(dtype=float),
                descriptor.predictions,
                full.predictions,
                cast(pd.Series, target_data["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {direction} | {label} | {descriptor.r2:.3f} | {full.r2:.3f} | "
                f"{difference.delta_r2:.3f} | "
                f"[{difference.lower:.3f}, {difference.upper:.3f}] | "
                f"{descriptor.mae:.3f} | {full.mae:.3f} | {full.correlation:.3f} |"
                f" {full.spearman:.3f} |"
            )

    if {"speaker_id", "cr_turn_composite"}.issubset(scored.columns):
        trajectories = build_metric_trajectories(scored)
        dialogue_groups = cast(
            pd.DataFrame,
            scored.loc[:, ["dialogue_id", "group"]].drop_duplicates("dialogue_id"),
        )
        trajectories = trajectories.merge(
            dialogue_groups,
            on="dialogue_id",
            how="left",
            validate="many_to_one",
        )
        trajectories["group_ai"] = (trajectories["group"] == "AI").astype(int)
        trajectory_metrics = [
            "surprisal_mean",
            "surprisal_sent_max",
            "sem_dist_partner",
        ]
        controls = [
            *[f"{metric}_mean" for metric in trajectory_metrics],
            "mean_turn_length",
            "group_ai",
        ]
        ordered_features = [
            feature
            for metric in trajectory_metrics
            for feature in [f"{metric}_slope", f"{metric}_rise"]
        ]
        distribution_features = [
            feature
            for metric in trajectory_metrics
            for feature in [f"{metric}_variability", f"{metric}_peak"]
        ]
        trajectory_outcomes = {
            "creative_thinking": "Creative thinking",
            "originality": "Originality",
            "critical_thinking": "Critical thinking",
        }
        lines.extend(
            [
                "",
                "## Speaker-trajectory localization",
                "",
                (
                    f"The analysis includes {len(trajectories)} speaker trajectories from "
                    f"{trajectories['dialogue_id'].nunique()} dialogues. Dialogue is the "
                    "clustering and hold-out unit, and pooled models control interlocutor type. "
                    "Raw metric trajectories are constructed before fold-specific scaling."
                ),
                "",
                "| Outcome | Added feature block | Delta R2 | Cluster-robust p | BH q |",
                "|---|---|---:|---:|---:|",
            ]
        )
        trajectory_tests: list[tuple[str, str, ClusteredIncrement]] = []
        for outcome, label in trajectory_outcomes.items():
            all_result = clustered_linear_incremental_test(
                trajectories,
                outcome,
                controls,
                [*ordered_features, *distribution_features],
            )
            ordered_result = clustered_linear_incremental_test(
                trajectories,
                outcome,
                [*controls, *distribution_features],
                ordered_features,
            )
            distribution_result = clustered_linear_incremental_test(
                trajectories,
                outcome,
                [*controls, *ordered_features],
                distribution_features,
            )
            for block, result in [
                ("All four descriptors", all_result),
                ("Ordered slope and rise", ordered_result),
                ("Variability and peak", distribution_result),
            ]:
                trajectory_tests.append((label, block, result))
        trajectory_q = benjamini_hochberg([result.p for _label, _block, result in trajectory_tests])
        for (label, block, result), q_value in zip(
            trajectory_tests,
            trajectory_q,
            strict=True,
        ):
            lines.append(
                f"| {label} | {block} | {result.delta_r2:.3f} | "
                f"{_format_p(result.p)} | {_format_p(q_value)} |"
            )

        lines.extend(
            [
                "",
                "| Outcome | Mean-level held-out R2 | Full-trajectory held-out R2 | "
                "Delta R2 | 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for outcome, label in trajectory_outcomes.items():
            baseline = held_out_dialogue_evaluation(
                trajectories,
                outcome,
                controls,
                outer_splits=5,
                inner_splits=4,
                repeats=cv_repeats,
                strata="group",
            )
            full = held_out_dialogue_evaluation(
                trajectories,
                outcome,
                [*controls, *ordered_features, *distribution_features],
                outer_splits=5,
                inner_splits=4,
                repeats=cv_repeats,
                strata="group",
            )
            difference = paired_group_bootstrap_delta_r2(
                cast(pd.Series, trajectories[outcome]).to_numpy(dtype=float),
                baseline.predictions,
                full.predictions,
                cast(pd.Series, trajectories["dialogue_id"]).to_numpy(),
                replicates=bootstrap_replicates,
            )
            lines.append(
                f"| {label} | {baseline.r2:.3f} | {full.r2:.3f} | "
                f"{difference.delta_r2:.3f} | "
                f"[{difference.lower:.3f}, {difference.upper:.3f}] |"
            )

    return "\n".join(lines) + "\n"


def write_robust_validation_report(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    bootstrap_replicates: int = 1_000,
    cv_repeats: int = 5,
    tables_directory: str | Path | None = None,
) -> Path:
    """Write the BRM-oriented validation report and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = robust_validation_report(
        frame,
        bootstrap_replicates=bootstrap_replicates,
        cv_repeats=cv_repeats,
    )
    path.write_text(report, encoding="utf-8")
    if tables_directory is not None:
        directory = Path(tables_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in extract_markdown_tables(report).items():
            table.to_csv(directory / f"{name}.csv", index=False)
    return path
