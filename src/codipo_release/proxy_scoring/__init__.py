"""Image-grounded coverage and residual-similarity proxy scoring."""

from codipo_release.proxy_scoring.core import (
    ProxySettings,
    build_class_statistics,
    candidate_bound_check,
    compute_dynamic_bounds,
    compute_yz,
    estimate_residual_projection,
    residual_similarity,
)

__all__ = [
    "ProxySettings",
    "build_class_statistics",
    "candidate_bound_check",
    "compute_dynamic_bounds",
    "compute_yz",
    "estimate_residual_projection",
    "residual_similarity",
]
