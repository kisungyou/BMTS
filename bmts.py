"""Core routines for Bayesian mixture transport signatures (BMTS).

This module is the small public computational surface used by the notebooks.
It contains no experiment-level execution and performs no file writes on
import.  The finite-metric dual implementation is exact up to floating-point
arithmetic for the small component dictionaries used in the paper.
"""

from __future__ import annotations

import itertools
import inspect
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Small linear-algebra calls are more reproducible and usually faster here
# when BLAS does not start a large worker pool.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "bmts-public-matplotlib-cache"),
)

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.optimize import linprog
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


__all__ = [
    "MixtureDraws",
    "PosteriorBMTS",
    "sample_isotropic_gmm",
    "sample_common_covariance_gmm",
    "gaussian_w2_ground_cost",
    "bmts_matrix",
    "bmts_pairs",
    "optimal_transport_plan",
    "posterior_bmts",
    "residual_aware_bmts",
    "residual_aware_matrix",
    "rao_blackwell_psm",
    "responsibility_entropy",
    "pooled_within_covariance",
    "largest_gap_threshold",
    "percentile_rank",
    "classical_mds",
    "neighbor_pairs",
    "local_pair_medians",
    "facs_neighborhood_mixing",
    "load_nestorowa",
    "set_notebook_style",
    "panel_label",
    "finish_axes",
    "draw_relation_graph",
    "draw_violin_summary",
    "run_self_checks",
]


MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"

TEXT_WIDTH_IN = 6.5
INK = "#1F2933"
MUTED = "#5F6B76"
GRID = "#D9DEE3"
LIGHT_GRID = "#EEF1F3"
PAPER = "#FFFFFF"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#6F4E7C"
SKY = "#56B4E9"
BLACK = "#000000"
CATEGORY_COLORS = (BLUE, ORANGE, GREEN, PURPLE, VERMILLION, SKY)
CATEGORY_MARKERS = ("o", "s", "^", "D", "P", "v")

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "bmts_sequential",
    ("#F7FAFC", "#C9E2EA", "#79B4C3", "#357C91", "#173B57"),
)
UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list(
    "bmts_uncertainty",
    ("#FCFAF2", "#F3D58A", "#DF8F44", "#A84A5B", "#4B284F"),
)

PAPER_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.facecolor": PAPER,
    "savefig.edgecolor": "none",
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8.25,
    "mathtext.fontset": "cm",
    "axes.labelsize": 8.25,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.65,
    "axes.prop_cycle": cycler(
        color=(BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY)
    ),
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7.4,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 0.9,
    "lines.linewidth": 1.25,
    "lines.markersize": 4.0,
    "patch.linewidth": 0.65,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclass(frozen=True)
class MixtureDraws:
    """Posterior draws from a finite Gaussian mixture with fixed covariance."""

    weights: np.ndarray
    means: np.ndarray
    allocations: np.ndarray
    responsibilities: np.ndarray

    @property
    def n_draws(self) -> int:
        return int(self.weights.shape[0])


@dataclass(frozen=True)
class PosteriorBMTS:
    """Posterior BMTS draws and pointwise interval summaries."""

    draws: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray


def _softmax_rows(log_probability: np.ndarray) -> np.ndarray:
    centered = log_probability - logsumexp(
        log_probability, axis=1, keepdims=True
    )
    return np.exp(centered)


def _sample_categorical_rows(
    probabilities: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    uniforms = rng.random(probabilities.shape[0])[:, None]
    return (uniforms > cumulative[:, :-1]).sum(axis=1)


def _log_density_isotropic(
    observations: np.ndarray, means: np.ndarray, variance: float
) -> np.ndarray:
    differences = observations[:, None, :] - means[None, :, :]
    dimension = observations.shape[1]
    return -0.5 * (
        np.sum(differences * differences, axis=2) / variance
        + dimension * np.log(2.0 * np.pi * variance)
    )


def _log_density_common_covariance(
    observations: np.ndarray,
    means: np.ndarray,
    inverse: np.ndarray,
    log_determinant: float,
) -> np.ndarray:
    differences = observations[:, None, :] - means[None, :, :]
    quadratic = np.einsum(
        "nki,ij,nkj->nk", differences, inverse, differences
    )
    dimension = observations.shape[1]
    return -0.5 * (
        quadratic + dimension * np.log(2.0 * np.pi) + log_determinant
    )


def _validate_sampler_settings(
    observations: np.ndarray,
    components: int,
    iterations: int,
    burn: int,
    thin: int,
) -> np.ndarray:
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or len(observations) < 2:
        raise ValueError("observations must be a two-dimensional array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")
    if components < 1 or components > len(observations):
        raise ValueError("components must lie between one and the sample size")
    if not (0 <= burn < iterations) or thin < 1:
        raise ValueError("require 0 <= burn < iterations and thin >= 1")
    return observations


def sample_isotropic_gmm(
    observations: np.ndarray,
    components: int,
    variance: float,
    *,
    iterations: int = 1_800,
    burn: int = 600,
    thin: int = 4,
    concentration: float = 1.0,
    prior_variance: float = 100.0,
    seed: int = 2_307,
) -> MixtureDraws:
    """Gibbs sample a finite GMM with fixed common covariance ``variance I``."""

    observations = _validate_sampler_settings(
        observations, components, iterations, burn, thin
    )
    if variance <= 0 or concentration <= 0 or prior_variance <= 0:
        raise ValueError("variance, concentration, and prior_variance must be positive")

    rng = np.random.default_rng(seed)
    n, dimension = observations.shape
    initial = KMeans(
        n_clusters=components, n_init=20, random_state=seed
    ).fit(observations)
    allocations = initial.labels_.copy()
    means = np.vstack(
        [
            observations[allocations == k].mean(axis=0)
            if np.any(allocations == k)
            else observations[rng.integers(n)]
            for k in range(components)
        ]
    )
    counts = np.bincount(allocations, minlength=components)
    weights = rng.dirichlet(concentration / components + counts)
    prior_mean = observations.mean(axis=0)

    weight_draws: list[np.ndarray] = []
    mean_draws: list[np.ndarray] = []
    allocation_draws: list[np.ndarray] = []
    responsibility_draws: list[np.ndarray] = []

    for iteration in range(iterations):
        log_probability = np.log(weights + 1e-300)[None, :] + _log_density_isotropic(
            observations, means, variance
        )
        probabilities = _softmax_rows(log_probability)
        allocations = _sample_categorical_rows(probabilities, rng)
        counts = np.bincount(allocations, minlength=components)
        weights = rng.dirichlet(concentration / components + counts)

        for k in range(components):
            subset = observations[allocations == k]
            precision = 1.0 / prior_variance + len(subset) / variance
            posterior_variance = 1.0 / precision
            numerator = prior_mean / prior_variance
            if len(subset):
                numerator = numerator + subset.sum(axis=0) / variance
            posterior_mean = posterior_variance * numerator
            means[k] = rng.normal(
                posterior_mean,
                np.sqrt(posterior_variance),
                size=dimension,
            )

        if iteration >= burn and (iteration - burn) % thin == 0:
            log_probability = np.log(weights + 1e-300)[None, :] + _log_density_isotropic(
                observations, means, variance
            )
            weight_draws.append(weights.copy())
            mean_draws.append(means.copy())
            allocation_draws.append(allocations.copy())
            responsibility_draws.append(_softmax_rows(log_probability))

    return MixtureDraws(
        weights=np.asarray(weight_draws),
        means=np.asarray(mean_draws),
        allocations=np.asarray(allocation_draws),
        responsibilities=np.asarray(responsibility_draws),
    )


def sample_common_covariance_gmm(
    observations: np.ndarray,
    components: int,
    covariance: np.ndarray,
    *,
    iterations: int = 1_500,
    burn: int = 600,
    thin: int = 3,
    concentration: float = 1.0,
    prior_scale: float = 25.0,
    seed: int = 5_179,
    initial_labels: np.ndarray | None = None,
) -> MixtureDraws:
    """Gibbs sample a finite GMM with a fixed full common covariance.

    Component means have prior ``N(sample_mean, prior_scale * covariance)``.
    """

    observations = _validate_sampler_settings(
        observations, components, iterations, burn, thin
    )
    if concentration <= 0 or prior_scale <= 0:
        raise ValueError("concentration and prior_scale must be positive")
    covariance = np.asarray(covariance, dtype=float)
    dimension = observations.shape[1]
    if covariance.shape != (dimension, dimension):
        raise ValueError("covariance has an incompatible shape")
    covariance = 0.5 * (covariance + covariance.T)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    inverse = np.linalg.inv(covariance)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")

    rng = np.random.default_rng(seed)
    n = len(observations)
    if initial_labels is None:
        allocations = KMeans(
            n_clusters=components, n_init=20, random_state=seed
        ).fit_predict(observations)
    else:
        allocations = np.asarray(initial_labels, dtype=int).copy()
        if allocations.shape != (n,):
            raise ValueError("initial_labels must contain one label per observation")
        if np.any(allocations < 0) or np.any(allocations >= components):
            raise ValueError("initial_labels contain an invalid component")

    means = np.vstack(
        [
            observations[allocations == k].mean(axis=0)
            if np.any(allocations == k)
            else observations[rng.integers(n)]
            for k in range(components)
        ]
    )
    counts = np.bincount(allocations, minlength=components)
    weights = rng.dirichlet(concentration / components + counts)
    prior_mean = observations.mean(axis=0)

    weight_draws: list[np.ndarray] = []
    mean_draws: list[np.ndarray] = []
    allocation_draws: list[np.ndarray] = []
    responsibility_draws: list[np.ndarray] = []

    for iteration in range(iterations):
        log_probability = np.log(weights + 1e-300)[None, :] + _log_density_common_covariance(
            observations, means, inverse, log_determinant
        )
        allocations = _sample_categorical_rows(
            _softmax_rows(log_probability), rng
        )
        counts = np.bincount(allocations, minlength=components)
        weights = rng.dirichlet(concentration / components + counts)

        for k in range(components):
            subset = observations[allocations == k]
            kappa = len(subset) + 1.0 / prior_scale
            numerator = prior_mean / prior_scale
            if len(subset):
                numerator = numerator + subset.sum(axis=0)
            posterior_mean = numerator / kappa
            means[k] = posterior_mean + rng.normal(size=dimension) @ (
                cholesky / np.sqrt(kappa)
            ).T

        if iteration >= burn and (iteration - burn) % thin == 0:
            log_probability = np.log(weights + 1e-300)[None, :] + _log_density_common_covariance(
                observations, means, inverse, log_determinant
            )
            weight_draws.append(weights.copy())
            mean_draws.append(means.copy())
            allocation_draws.append(allocations.copy())
            responsibility_draws.append(_softmax_rows(log_probability))

    return MixtureDraws(
        weights=np.asarray(weight_draws),
        means=np.asarray(mean_draws),
        allocations=np.asarray(allocation_draws),
        responsibilities=np.asarray(responsibility_draws),
    )


def gaussian_w2_ground_cost(component_means: np.ndarray) -> np.ndarray:
    """Gaussian Wasserstein ground cost when all covariances are common."""

    component_means = np.asarray(component_means, dtype=float)
    if component_means.ndim != 2:
        raise ValueError("component_means must be a K by d array")
    return np.linalg.norm(
        component_means[:, None, :] - component_means[None, :, :], axis=2
    )


def _metric_dual_vertices(cost: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Enumerate finite-metric Kantorovich--Rubinstein dual vertices."""

    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be square")
    if not np.all(np.isfinite(cost)) or np.any(cost < -tolerance):
        raise ValueError("cost must be finite and nonnegative")
    if not np.allclose(cost, cost.T, atol=tolerance, rtol=0):
        raise ValueError("the dual method requires a symmetric metric cost")
    if not np.allclose(np.diag(cost), 0.0, atol=tolerance, rtol=0):
        raise ValueError("a metric cost must have zero diagonal")
    if np.any(
        cost[:, :, None]
        > cost[:, None, :] + cost[None, :, :] + tolerance
    ):
        raise ValueError("the dual method requires the triangle inequality")

    components = cost.shape[0]
    if components == 1:
        return np.zeros((1, 1))

    rows: list[np.ndarray] = []
    bounds: list[float] = []
    for i in range(components):
        for j in range(components):
            if i == j:
                continue
            row = np.zeros(components)
            row[i] = 1.0
            row[j] = -1.0
            rows.append(row)
            bounds.append(float(cost[i, j]))
    inequalities = np.vstack(rows)
    bound_array = np.asarray(bounds)
    gauge = np.zeros(components)
    gauge[0] = 1.0

    vertices: list[np.ndarray] = []
    for active in itertools.combinations(
        range(len(bounds)), components - 1
    ):
        equations = np.vstack([gauge, inequalities[list(active)]])
        right_side = np.r_[0.0, bound_array[list(active)]]
        try:
            vertex = np.linalg.solve(equations, right_side)
        except np.linalg.LinAlgError:
            continue
        if np.all(inequalities @ vertex <= bound_array + tolerance):
            vertices.append(vertex)
    if not vertices:
        raise RuntimeError("no feasible metric dual vertices were found")
    candidate = np.vstack(vertices)
    _, unique = np.unique(
        np.round(candidate, 10), axis=0, return_index=True
    )
    return candidate[np.sort(unique)]


def _validate_responsibilities(
    responsibilities: np.ndarray, tolerance: float = 1e-8
) -> np.ndarray:
    responsibilities = np.asarray(responsibilities, dtype=float)
    if responsibilities.ndim != 2:
        raise ValueError("responsibilities must be an n by K array")
    if not np.all(np.isfinite(responsibilities)) or np.any(
        responsibilities < -tolerance
    ):
        raise ValueError("responsibilities must be finite and nonnegative")
    if not np.allclose(
        responsibilities.sum(axis=1), 1.0, atol=tolerance, rtol=0
    ):
        raise ValueError("each responsibility vector must sum to one")
    return responsibilities


def bmts_matrix(
    responsibilities: np.ndarray,
    ground_cost: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Compute exact all-pairs BMTS values on a common finite metric support."""

    responsibilities = _validate_responsibilities(
        responsibilities, tolerance
    )
    vertices = _metric_dual_vertices(ground_cost, tolerance)
    if vertices.shape[1] != responsibilities.shape[1]:
        raise ValueError("responsibilities and ground_cost disagree on K")
    scores = responsibilities @ vertices.T
    distances = np.zeros((len(responsibilities), len(responsibilities)))
    for column in range(vertices.shape[0]):
        differences = scores[:, [column]] - scores[:, [column]].T
        distances = np.maximum(distances, np.abs(differences))
    np.fill_diagonal(distances, 0.0)
    return distances


def bmts_pairs(
    responsibilities: np.ndarray,
    ground_cost: np.ndarray,
    pairs: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Compute exact BMTS values for selected observation pairs."""

    responsibilities = _validate_responsibilities(
        responsibilities, tolerance
    )
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be an m by 2 array")
    if np.any(pairs < 0) or np.any(pairs >= len(responsibilities)):
        raise ValueError("pairs contain an invalid observation index")
    vertices = _metric_dual_vertices(ground_cost, tolerance)
    if vertices.shape[1] != responsibilities.shape[1]:
        raise ValueError("responsibilities and ground_cost disagree on K")
    scores = responsibilities @ vertices.T
    return np.max(
        np.abs(scores[pairs[:, 0]] - scores[pairs[:, 1]]), axis=1
    )


def optimal_transport_plan(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return one exact primal transport plan for two probability vectors."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    cost = np.asarray(cost, dtype=float)
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("source and target must be equally sized vectors")
    components = len(source)
    if cost.shape != (components, components):
        raise ValueError("cost must be K by K")
    if (
        not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(cost))
        or np.any(source < -tolerance)
        or np.any(target < -tolerance)
        or np.any(cost < -tolerance)
    ):
        raise ValueError("marginals and costs must be finite and nonnegative")
    if not np.isclose(source.sum(), 1.0, atol=tolerance, rtol=0) or not np.isclose(
        target.sum(), 1.0, atol=tolerance, rtol=0
    ):
        raise ValueError("source and target must each sum to one")

    equalities: list[np.ndarray] = []
    right_side: list[float] = []
    for i in range(components):
        row = np.zeros(components * components)
        row[i * components : (i + 1) * components] = 1.0
        equalities.append(row)
        right_side.append(float(source[i]))
    for j in range(components):
        row = np.zeros(components * components)
        row[j::components] = 1.0
        equalities.append(row)
        right_side.append(float(target[j]))

    result = linprog(
        cost.ravel(),
        A_eq=np.asarray(equalities),
        b_eq=np.asarray(right_side),
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.reshape(components, components)


def posterior_bmts(
    mean_draws: np.ndarray,
    responsibility_draws: np.ndarray,
    *,
    pairs: np.ndarray | None = None,
    quantiles: tuple[float, float] = (0.05, 0.95),
    dtype: np.dtype | type = np.float64,
) -> PosteriorBMTS:
    """Evaluate BMTS draw by draw and summarize a pointwise interval."""

    mean_draws = np.asarray(mean_draws, dtype=float)
    responsibility_draws = np.asarray(responsibility_draws, dtype=float)
    if mean_draws.ndim != 3 or responsibility_draws.ndim != 3:
        raise ValueError("mean_draws and responsibility_draws must be 3D")
    if mean_draws.shape[0] != responsibility_draws.shape[0]:
        raise ValueError("mean and responsibility draw counts differ")
    if mean_draws.shape[1] != responsibility_draws.shape[2]:
        raise ValueError("mean and responsibility component counts differ")
    if not (0 <= quantiles[0] < quantiles[1] <= 1):
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")

    if pairs is None:
        n = responsibility_draws.shape[1]
        values = np.empty((len(mean_draws), n, n), dtype=dtype)
    else:
        pairs = np.asarray(pairs, dtype=int)
        values = np.empty((len(mean_draws), len(pairs)), dtype=dtype)

    for draw_index, (means, responsibilities) in enumerate(
        zip(mean_draws, responsibility_draws)
    ):
        cost = gaussian_w2_ground_cost(means)
        if pairs is None:
            values[draw_index] = bmts_matrix(responsibilities, cost)
        else:
            values[draw_index] = bmts_pairs(
                responsibilities, cost, pairs
            )

    lower, upper = np.quantile(values, quantiles, axis=0)
    return PosteriorBMTS(
        draws=values,
        mean=values.mean(axis=0),
        lower=lower,
        upper=upper,
        width=upper - lower,
    )


def residual_aware_bmts(
    source: np.ndarray,
    target: np.ndarray,
    ground_cost: np.ndarray,
    source_residuals: np.ndarray,
    target_residuals: np.ndarray,
    residual_weight: float,
) -> float:
    """Compute the pair-specific residual-aware BMTS dissimilarity."""

    if residual_weight < 0:
        raise ValueError("residual_weight must be nonnegative")
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_residuals = np.asarray(source_residuals, dtype=float)
    target_residuals = np.asarray(target_residuals, dtype=float)
    components = len(source)
    if (
        target.shape != source.shape
        or source_residuals.shape != target_residuals.shape
        or source_residuals.ndim != 2
        or source_residuals.shape[0] != components
    ):
        raise ValueError("residual arrays must have common shape K by d")
    pair_cost = np.asarray(ground_cost, dtype=float).copy()
    if pair_cost.shape != (components, components):
        raise ValueError("ground_cost must be K by K")
    for k in range(components):
        pair_cost[k, k] += residual_weight * np.linalg.norm(
            source_residuals[k] - target_residuals[k]
        )
    plan = optimal_transport_plan(source, target, pair_cost)
    return float(np.sum(plan * pair_cost))


def residual_aware_matrix(
    responsibilities: np.ndarray,
    ground_cost: np.ndarray,
    residuals: np.ndarray,
    residual_weight: float,
) -> np.ndarray:
    """Compute all-pairs residual-aware BMTS values with the primal LP."""

    responsibilities = _validate_responsibilities(responsibilities)
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 3 or residuals.shape[:2] != responsibilities.shape:
        raise ValueError("residuals must have shape n by K by d")
    n = len(responsibilities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            value = residual_aware_bmts(
                responsibilities[i],
                responsibilities[j],
                ground_cost,
                residuals[i],
                residuals[j],
                residual_weight,
            )
            distances[i, j] = distances[j, i] = value
    return distances


def rao_blackwell_psm(responsibility_draws: np.ndarray) -> np.ndarray:
    """Estimate co-clustering probabilities from conditional responsibilities."""

    responsibility_draws = np.asarray(responsibility_draws, dtype=float)
    if responsibility_draws.ndim != 3:
        raise ValueError("responsibility_draws must have shape M by n by K")
    if not np.all(np.isfinite(responsibility_draws)) or np.any(
        responsibility_draws < -1e-8
    ):
        raise ValueError("responsibilities must be finite and nonnegative")
    if not np.allclose(
        responsibility_draws.sum(axis=2), 1.0, atol=1e-8, rtol=0
    ):
        raise ValueError("each responsibility vector must sum to one")
    psm = np.einsum(
        "mik,mjk->ij", responsibility_draws, responsibility_draws
    ) / len(responsibility_draws)
    np.fill_diagonal(psm, 1.0)
    return psm


def responsibility_entropy(
    responsibility_draws: np.ndarray, *, normalized: bool = True
) -> np.ndarray:
    """Posterior mean observation-level responsibility entropy."""

    responsibility_draws = np.asarray(responsibility_draws, dtype=float)
    if responsibility_draws.ndim != 3:
        raise ValueError("responsibility_draws must have shape M by n by K")
    components = responsibility_draws.shape[2]
    entropy = -np.sum(
        responsibility_draws * np.log(responsibility_draws + 1e-300), axis=2
    ).mean(axis=0)
    if normalized and components > 1:
        entropy = entropy / np.log(components)
    return entropy


def pooled_within_covariance(
    observations: np.ndarray, labels: np.ndarray, *, ridge: float = 0.0
) -> np.ndarray:
    """Estimate a pooled within-group covariance from provisional labels."""

    observations = np.asarray(observations, dtype=float)
    labels = np.asarray(labels)
    if observations.ndim != 2 or labels.shape != (len(observations),):
        raise ValueError("labels must contain one entry per observation")
    unique = np.unique(labels)
    scatter = np.zeros((observations.shape[1], observations.shape[1]))
    for level in unique:
        subset = observations[labels == level]
        if len(subset):
            centered = subset - subset.mean(axis=0)
            scatter += centered.T @ centered
    denominator = len(observations) - len(unique)
    if denominator <= 0:
        raise ValueError("not enough residual degrees of freedom")
    covariance = scatter / denominator
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    return covariance + ridge * np.eye(observations.shape[1])


def largest_gap_threshold(
    values: np.ndarray,
) -> tuple[float, float, float, float]:
    """Choose the midpoint of the largest gap in sorted values."""

    values = np.sort(np.asarray(values, dtype=float))
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two one-dimensional values are required")
    gaps = np.diff(values)
    index = int(np.argmax(gaps))
    return (
        float((values[index] + values[index + 1]) / 2.0),
        float(values[index]),
        float(values[index + 1]),
        float(gaps[index]),
    )


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile ranks on the zero-to-one scale."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(values) <= 1:
        return np.zeros(len(values))
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def classical_mds(distances: np.ndarray, dimensions: int = 2) -> np.ndarray:
    """Classical multidimensional scaling for a symmetric distance matrix."""

    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be square")
    n = len(distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.maximum(eigenvalues[order[:dimensions]], 0.0)
    return eigenvectors[:, order[:dimensions]] * np.sqrt(positive)


def neighbor_pairs(
    observations: np.ndarray, neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed neighbor rows and their undirected pair union."""

    observations = np.asarray(observations, dtype=float)
    if not (1 <= neighbors < len(observations)):
        raise ValueError("neighbors must lie between one and n - 1")
    queried = NearestNeighbors(n_neighbors=neighbors + 1).fit(
        observations
    ).kneighbors(observations, return_distance=False)
    rows = np.empty((len(observations), neighbors), dtype=int)
    for i, row in enumerate(queried):
        rows[i] = row[row != i][:neighbors]
    pairs = np.asarray(
        sorted(
            {
                tuple(sorted((i, int(j))))
                for i, row in enumerate(rows)
                for j in row
            }
        ),
        dtype=int,
    )
    return rows, pairs


def local_pair_medians(
    pair_values: np.ndarray,
    neighbor_rows: np.ndarray,
    pairs: np.ndarray,
    *,
    neighbors: int | None = None,
) -> np.ndarray:
    """Summarize pair values by each cell's nearest-neighbor median."""

    pair_values = np.asarray(pair_values, dtype=float)
    neighbor_rows = np.asarray(neighbor_rows, dtype=int)
    pairs = np.asarray(pairs, dtype=int)
    if pair_values.shape != (len(pairs),):
        raise ValueError("pair_values must contain one value per pair")
    if neighbors is None:
        neighbors = neighbor_rows.shape[1]
    if not (1 <= neighbors <= neighbor_rows.shape[1]):
        raise ValueError("neighbors is outside the available neighbor rows")
    lookup = {tuple(pair): index for index, pair in enumerate(pairs)}
    output = np.empty(len(neighbor_rows))
    for i, row in enumerate(neighbor_rows[:, :neighbors]):
        indices = [lookup[tuple(sorted((i, int(j))))] for j in row]
        output[i] = np.median(pair_values[indices])
    return output


def facs_neighborhood_mixing(
    facs: np.ndarray,
    fine_labels: np.ndarray,
    levels: Iterable[str],
    *,
    neighbors: int = 30,
) -> np.ndarray:
    """Withheld FACS boundary score based on conventionally labeled cells."""

    facs = np.asarray(facs, dtype=float)
    fine_labels = np.asarray(fine_labels).astype("U64")
    levels = np.asarray(list(levels)).astype("U64")
    if facs.ndim != 2 or fine_labels.shape != (len(facs),):
        raise ValueError("fine_labels must contain one entry per FACS row")
    observed = np.all(np.isfinite(facs), axis=1)
    scaled = np.full_like(facs, np.nan)
    scaled[observed] = StandardScaler().fit_transform(facs[observed])
    known = fine_labels != ""
    reference = known & observed
    reference_indices = np.flatnonzero(reference)
    reference_labels = fine_labels[reference]
    if neighbors >= len(reference_indices):
        raise ValueError("neighbors is too large for the labeled reference set")

    model = NearestNeighbors(n_neighbors=neighbors + 1).fit(scaled[reference])
    observed_indices = np.flatnonzero(observed)
    queried = model.kneighbors(scaled[observed], return_distance=False)
    score = np.full(len(facs), np.nan)
    for cell, row in zip(observed_indices, queried):
        global_neighbors = reference_indices[row]
        if reference[cell]:
            row = row[global_neighbors != cell][:neighbors]
        else:
            row = row[:neighbors]
        proportions = [np.mean(reference_labels[row] == level) for level in levels]
        score[cell] = 1.0 - max(proportions)
    return score


def load_nestorowa(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load the compact, pickle-free Nestorowa HSPC derivative."""

    source = DATA_DIR / "nestorowa_bmts_input.npz" if path is None else Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Nestorowa derivative not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    if data.get("rna_pcs", np.empty((0, 0))).shape != (1_656, 9):
        raise ValueError("expected 1,656 cells and nine RNA principal components")
    return data


def set_notebook_style() -> None:
    """Apply the manuscript-aligned, TeX-free notebook rcParams."""

    plt.rcParams.update(PAPER_RCPARAMS)


def panel_label(ax, label: str) -> None:
    """Place a consistent panel label above an axes frame."""

    ax.text(
        -0.085,
        1.035,
        label,
        transform=ax.transAxes,
        color=BLACK,
        fontsize=9.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def finish_axes(ax, grid_axis: str | None = None) -> None:
    """Apply the shared light-touch axis treatment."""

    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=LIGHT_GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def draw_relation_graph(
    ax,
    positions: np.ndarray,
    adjacency: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    group_names: Iterable[str] | None = None,
    edge_values: np.ndarray | None = None,
    edge_alpha: float = 0.08,
    node_size: float = 16.0,
) -> None:
    """Draw a dense undirected relation graph without a network dependency."""

    positions = np.asarray(positions, dtype=float)
    adjacency = np.asarray(adjacency)
    edge_index = np.column_stack(np.nonzero(np.triu(adjacency, k=1)))
    if edge_index.size:
        segments = np.stack(
            [positions[edge_index[:, 0]], positions[edge_index[:, 1]]], axis=1
        )
        if edge_values is None:
            colors = MUTED
            widths = 0.28
        else:
            strengths = np.asarray(edge_values)[
                edge_index[:, 0], edge_index[:, 1]
            ]
            scaled = np.clip((strengths - 0.8) / 0.2, 0.0, 1.0)
            colors = [
                to_rgba(MUTED, 0.01 + edge_alpha * (0.08 + 0.22 * value))
                for value in scaled
            ]
            widths = 0.16 + 0.22 * scaled
        ax.add_collection(
            LineCollection(
                segments,
                colors=colors,
                linewidths=widths,
                alpha=edge_alpha if edge_values is None else None,
                rasterized=True,
                zorder=1,
            )
        )

    if groups is None:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=node_size,
            color=BLUE,
            alpha=0.88,
            edgecolor=PAPER,
            linewidth=0.35,
            zorder=2,
        )
    else:
        groups = np.asarray(groups)
        names = list(group_names) if group_names is not None else None
        for group in np.unique(groups):
            mask = groups == group
            label = names[int(group)] if names is not None else str(group)
            ax.scatter(
                positions[mask, 0],
                positions[mask, 1],
                s=node_size,
                color=CATEGORY_COLORS[int(group) % len(CATEGORY_COLORS)],
                marker=CATEGORY_MARKERS[int(group) % len(CATEGORY_MARKERS)],
                alpha=0.9,
                edgecolor=PAPER,
                linewidth=0.4,
                label=label,
                zorder=2,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.margins(x=0.07, y=0.10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)


def draw_violin_summary(
    ax,
    values: Iterable[np.ndarray],
    colors: Iterable[str],
    *,
    seed: int,
    max_points: int = 150,
    orientation: str = "vertical",
) -> None:
    """Draw restrained violins with median/IQR bars and light points."""

    values = [np.asarray(value, dtype=float) for value in values]
    colors = list(colors)
    positions = np.arange(1, len(values) + 1)
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be vertical or horizontal")
    orientation_argument = (
        {"orientation": orientation}
        if "orientation" in inspect.signature(ax.violinplot).parameters
        else {"vert": orientation == "vertical"}
    )
    parts = ax.violinplot(
        values,
        positions=positions,
        widths=0.76,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
        **orientation_argument,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
        body.set_linewidth(0.75)

    rng = np.random.default_rng(seed)
    for position, values_here, color in zip(positions, values, colors):
        indices = np.linspace(
            0,
            len(values_here) - 1,
            min(max_points, len(values_here)),
            dtype=int,
        )
        jitter = rng.uniform(-0.065, 0.065, size=len(indices))
        q1, median, q3 = np.quantile(values_here, (0.25, 0.5, 0.75))
        if orientation == "vertical":
            ax.scatter(
                position + jitter,
                values_here[indices],
                s=5.0,
                color=color,
                alpha=0.20,
                edgecolor="none",
                zorder=2,
            )
            ax.vlines(position, q1, q3, color=color, linewidth=2.1, zorder=3)
            ax.hlines(
                median,
                position - 0.095,
                position + 0.095,
                color=INK,
                linewidth=1.0,
                zorder=4,
            )
            ax.scatter(
                position,
                values_here.mean(),
                marker="D",
                s=24,
                facecolor=PAPER,
                edgecolor=color,
                linewidth=1.0,
                zorder=5,
            )
        else:
            ax.scatter(
                values_here[indices],
                position + jitter,
                s=5.0,
                color=color,
                alpha=0.20,
                edgecolor="none",
                zorder=2,
            )
            ax.hlines(position, q1, q3, color=color, linewidth=2.1, zorder=3)
            ax.vlines(
                median,
                position - 0.095,
                position + 0.095,
                color=INK,
                linewidth=1.0,
                zorder=4,
            )
            ax.scatter(
                values_here.mean(),
                position,
                marker="D",
                s=24,
                facecolor=PAPER,
                edgecolor=color,
                linewidth=1.0,
                zorder=5,
            )
    finish_axes(ax, grid_axis="y" if orientation == "vertical" else "x")


def run_self_checks(seed: int = 99_173) -> dict[str, float]:
    """Run deterministic identities used to audit the public implementation."""

    rng = np.random.default_rng(seed)
    primal_dual_error = 0.0
    permutation_error = 0.0
    triangle_violation = 0.0
    for _ in range(12):
        component_points = rng.normal(size=(4, 2))
        cost = gaussian_w2_ground_cost(component_points)
        responsibilities = rng.dirichlet(np.ones(4), size=3)
        distances = bmts_matrix(responsibilities, cost)
        plan = optimal_transport_plan(
            responsibilities[0], responsibilities[1], cost
        )
        primal_dual_error = max(
            primal_dual_error,
            abs(distances[0, 1] - float(np.sum(plan * cost))),
        )
        triangle_violation = max(
            triangle_violation,
            float(distances[0, 2] - distances[0, 1] - distances[1, 2]),
        )
        permutation = rng.permutation(4)
        permuted = bmts_matrix(
            responsibilities[:, permutation],
            cost[np.ix_(permutation, permutation)],
        )
        permutation_error = max(
            permutation_error, float(np.max(np.abs(distances - permuted)))
        )

    one = np.ones((2, 1))
    residuals = np.asarray([[[0.2, -0.4]], [[1.1, 0.7]]])
    residual_value = residual_aware_bmts(
        one[0], one[1], np.zeros((1, 1)), residuals[0], residuals[1], 1.0
    )
    residual_closed = float(np.linalg.norm(residuals[0, 0] - residuals[1, 0]))

    draws = rng.dirichlet(np.ones(3), size=(20, 5))
    psm = rao_blackwell_psm(draws)
    return {
        "primal_dual_max_abs_error": float(primal_dual_error),
        "label_permutation_max_abs_error": float(permutation_error),
        "metric_triangle_max_positive_violation": float(
            max(0.0, triangle_violation)
        ),
        "residual_K1_reduction_abs_error": float(
            abs(residual_value - residual_closed)
        ),
        "psm_symmetry_max_abs_error": float(np.max(np.abs(psm - psm.T))),
        "psm_diagonal_max_abs_error": float(
            np.max(np.abs(np.diag(psm) - 1.0))
        ),
    }


if __name__ == "__main__":
    checks = run_self_checks()
    for name, value in checks.items():
        print(f"{name}: {value:.3e}")
