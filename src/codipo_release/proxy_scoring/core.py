"""Numerical definitions used by the CoDiPO X, Y and Z proxies.

The functions in this module are model independent. CLIP and GTE inference are
kept in separate entry points so that the definitions can be tested without
downloading model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class ProxySettings:
    """Locked proxy settings reported in the manuscript."""

    clip_threshold: float = 0.19
    chunk_weight: float = 0.8
    max_words_per_chunk: int = 8
    ad_residual_threshold: float = 0.14
    hc_residual_threshold: float = 0.16
    risk_band_multiplier: float = 0.4
    safe_band_multiplier: float = 1.0
    mad_consistency_factor: float = 1.4826

    def residual_threshold(self, label: int) -> float:
        if int(label) == 1:
            return self.ad_residual_threshold
        if int(label) == 0:
            return self.hc_residual_threshold
        raise ValueError(f"Label must be 0 or 1, found {label!r}.")


def fuse_clip_scores(
    global_scores: Sequence[float],
    chunk_scores: Sequence[float],
    *,
    chunk_weight: float = 0.8,
) -> np.ndarray:
    """Fuse full-transcript and best-chunk CLIP similarities."""

    if not 0.0 <= chunk_weight <= 1.0:
        raise ValueError("chunk_weight must be in [0, 1].")
    global_array = np.asarray(global_scores, dtype=float)
    chunk_array = np.asarray(chunk_scores, dtype=float)
    if global_array.shape != chunk_array.shape:
        raise ValueError("global_scores and chunk_scores must have the same shape.")
    return (1.0 - chunk_weight) * global_array + chunk_weight * chunk_array


def compute_yz(
    final_scores: Sequence[float],
    *,
    threshold: float = 0.19,
) -> dict[str, Any]:
    """Compute coverage breadth Y and activated-IU threshold-excess Z."""

    scores = np.asarray(final_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("final_scores must be one-dimensional.")
    total_count = int(scores.size)
    if total_count == 0:
        return {
            "Y": 0.0,
            "Z": 0.0,
            "hit_count": 0,
            "total_count": 0,
            "hit_mask": np.zeros(0, dtype=bool),
        }
    hit_mask = scores >= float(threshold)
    hit_count = int(hit_mask.sum())
    breadth = float(hit_count / total_count)
    depth = (
        float(np.mean(scores[hit_mask] - float(threshold)))
        if hit_count
        else 0.0
    )
    return {
        "Y": breadth,
        "Z": depth,
        "hit_count": hit_count,
        "total_count": total_count,
        "hit_mask": hit_mask,
    }


def _unit(vector: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError("AD and HC embedding means do not define a direction.")
    return vector / norm


def estimate_residual_projection(
    ad_embeddings: Sequence[Sequence[float]],
    hc_embeddings: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate global center and AD-minus-HC class-mean direction."""

    ad = np.asarray(ad_embeddings, dtype=float)
    hc = np.asarray(hc_embeddings, dtype=float)
    if ad.ndim != 2 or hc.ndim != 2 or ad.shape[1] != hc.shape[1]:
        raise ValueError("AD and HC embeddings must be compatible 2D arrays.")
    if not len(ad) or not len(hc):
        raise ValueError("Both AD and HC reference embeddings are required.")
    center = np.concatenate([ad, hc], axis=0).mean(axis=0)
    direction = _unit(ad.mean(axis=0) - hc.mean(axis=0))
    return center, direction


def residualize(
    embedding: Sequence[float],
    center: Sequence[float],
    direction: Sequence[float],
) -> np.ndarray:
    value = np.asarray(embedding, dtype=float) - np.asarray(center, dtype=float)
    unit_direction = _unit(np.asarray(direction, dtype=float))
    return value - np.dot(value, unit_direction) * unit_direction


def residual_similarity(
    source_embedding: Sequence[float],
    candidate_embedding: Sequence[float],
    center: Sequence[float],
    direction: Sequence[float],
    *,
    eps: float = 1e-12,
) -> float:
    """Cosine similarity after removing the AD-minus-HC mean direction."""

    source_residual = residualize(source_embedding, center, direction)
    candidate_residual = residualize(candidate_embedding, center, direction)
    source_norm = float(np.linalg.norm(source_residual))
    candidate_norm = float(np.linalg.norm(candidate_residual))
    if source_norm < 1e-6 or candidate_norm < 1e-6:
        return 1.0
    value = np.dot(source_residual, candidate_residual)
    return float(value / (source_norm * candidate_norm + eps))


def ordinary_cosine(
    first: Sequence[float],
    second: Sequence[float],
    *,
    eps: float = 1e-12,
) -> float:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first_array) * np.linalg.norm(second_array)
    return float(np.dot(first_array, second_array) / (denominator + eps))


def safe_mad(
    values: Sequence[float],
    *,
    consistency_factor: float = 1.4826,
    eps: float = 1e-6,
) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("MAD requires at least one value.")
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    return max(float(consistency_factor * mad), eps)


def build_class_statistics(
    y_values: Sequence[float],
    z_values: Sequence[float],
    *,
    consistency_factor: float = 1.4826,
) -> dict[str, float]:
    y_array = np.asarray(y_values, dtype=float)
    z_array = np.asarray(z_values, dtype=float)
    if y_array.shape != z_array.shape or y_array.size == 0:
        raise ValueError("Y and Z arrays must be non-empty and have equal shape.")
    return {
        "y_center": float(np.median(y_array)),
        "z_center": float(np.median(z_array)),
        "y_scale": safe_mad(
            y_array, consistency_factor=consistency_factor
        ),
        "z_scale": safe_mad(
            z_array, consistency_factor=consistency_factor
        ),
        "y_lower_risk": float(np.quantile(y_array, 0.10)),
        "y_upper_risk": float(np.quantile(y_array, 0.90)),
        "z_lower_risk": float(np.quantile(z_array, 0.10)),
        "z_upper_risk": float(np.quantile(z_array, 0.90)),
    }


def compute_dynamic_bounds(
    source_y: float,
    source_z: float,
    label: int,
    class_statistics: Mapping[str, float],
    *,
    risk_multiplier: float = 0.4,
    safe_multiplier: float = 1.0,
) -> dict[str, float]:
    """Construct diagnosis-directional source-centered Y/Z bands."""

    if int(label) not in {0, 1}:
        raise ValueError(f"Label must be 0 or 1, found {label!r}.")
    y_scale = float(class_statistics["y_scale"])
    z_scale = float(class_statistics["z_scale"])
    if int(label) == 1:
        y_plus, y_minus = risk_multiplier * y_scale, safe_multiplier * y_scale
        z_plus, z_minus = risk_multiplier * z_scale, safe_multiplier * z_scale
    else:
        y_plus, y_minus = safe_multiplier * y_scale, risk_multiplier * y_scale
        z_plus, z_minus = safe_multiplier * z_scale, risk_multiplier * z_scale
    return {
        "y_i": float(source_y),
        "z_i": float(source_z),
        "eps_y_plus": float(y_plus),
        "eps_y_minus": float(y_minus),
        "eps_z_plus": float(z_plus),
        "eps_z_minus": float(z_minus),
        "y_lower_i": float(source_y - y_minus),
        "y_upper_i": float(source_y + y_plus),
        "z_lower_i": float(source_z - z_minus),
        "z_upper_i": float(source_z + z_plus),
    }


def _directional_utilization_and_violation(
    delta: float,
    eps_plus: float,
    eps_minus: float,
) -> tuple[float, float, float]:
    eps_plus = max(float(eps_plus), EPS)
    eps_minus = max(float(eps_minus), EPS)
    denominator = eps_plus if delta >= 0 else eps_minus
    utilization = min(abs(float(delta)) / denominator, 1.0)
    violation_low = max(0.0, (-float(delta) - eps_minus) / eps_minus)
    violation_high = max(0.0, (float(delta) - eps_plus) / eps_plus)
    return utilization, violation_low, violation_high


def candidate_bound_check(
    candidate_y: float,
    candidate_z: float,
    bounds: Mapping[str, float],
) -> dict[str, Any]:
    delta_y = float(candidate_y - bounds["y_i"])
    delta_z = float(candidate_z - bounds["z_i"])
    y_util, y_low, y_high = _directional_utilization_and_violation(
        delta_y, bounds["eps_y_plus"], bounds["eps_y_minus"]
    )
    z_util, z_low, z_high = _directional_utilization_and_violation(
        delta_z, bounds["eps_z_plus"], bounds["eps_z_minus"]
    )
    violation = float(y_low + y_high + z_low + z_high)
    return {
        "yz_pass": violation == 0.0,
        "delta_y": delta_y,
        "delta_z": delta_z,
        "y_util": float(y_util),
        "z_util": float(z_util),
        "yz_utilization": float(0.5 * (y_util + z_util)),
        "y_violate_low": float(y_low),
        "y_violate_high": float(y_high),
        "z_violate_low": float(z_low),
        "z_violate_high": float(z_high),
        "yz_violation_score": violation,
        "y_lower_i": float(bounds["y_lower_i"]),
        "y_upper_i": float(bounds["y_upper_i"]),
        "z_lower_i": float(bounds["z_lower_i"]),
        "z_upper_i": float(bounds["z_upper_i"]),
    }


def evaluate_numeric_candidate(
    *,
    candidate_y: float,
    candidate_z: float,
    residual_cosine: float,
    origin_cosine: float,
    bounds: Mapping[str, float],
    residual_threshold: float,
) -> dict[str, Any]:
    """Assign the numeric fields used by source-local candidate ranking."""

    bound = candidate_bound_check(candidate_y, candidate_z, bounds)
    x_pass = float(residual_cosine) < float(residual_threshold)
    x_violation = max(
        0.0,
        (float(residual_cosine) - float(residual_threshold))
        / max(float(residual_threshold), EPS),
    )
    yz_pass = bool(bound["yz_pass"])
    if yz_pass and x_pass:
        bucket_id, bucket_name = 0, "yz_pass_x_pass"
    elif yz_pass:
        bucket_id, bucket_name = 1, "yz_pass_x_fail"
    elif x_pass:
        bucket_id, bucket_name = 2, "yz_fail_x_pass"
    else:
        bucket_id, bucket_name = 3, "yz_fail_x_fail"
    return {
        "candidate_y": float(candidate_y),
        "candidate_z": float(candidate_z),
        "residual_cos": float(residual_cosine),
        "origin_cos": float(origin_cosine),
        "cosine_threshold": float(residual_threshold),
        "x_pass": bool(x_pass),
        "x_violation_score": float(x_violation),
        "bucket_id": bucket_id,
        "bucket_name": bucket_name,
        **bound,
    }
