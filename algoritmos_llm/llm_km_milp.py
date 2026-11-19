"""
algoritmos_llm/llm_km_milp.py
=============================
Variante de KM-MILP guiada por prototipos semánticos Zero-Shot (Qwen / Gemma).
Reemplaza la inicialización aleatoria de centroides por los prototipos del LLM.
"""
from __future__ import annotations
import time
from typing import Any, Optional
import numpy as np
from scipy.optimize import Bounds
from scipy.spatial.distance import pdist, squareform

from algoritmos_base._common import compute_metrics, tabulate_clusters
from algoritmos_base.km_milp import _build_milp_constraints, _solve_milp_assignment


def run_llm_km_milp(
    x_mat: np.ndarray,
    y_true: np.ndarray,
    initial_centroids: np.ndarray,
    target_cardinality: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
    max_iter: int = 10,
    dist_method: str = "cosine",
    convergence_tol: float = 1e-4,
    time_limit: Optional[float] = 60.0,
) -> dict[str, Any]:
    """
    Ejecuta K-Means con asignación MILP partiendo de los prototipos del LLM.
    """
    x_mat = np.asarray(x_mat, dtype=float)
    centroids = np.asarray(initial_centroids, dtype=float).copy()
    target = np.asarray(target_cardinality, dtype=int)
    n, k = int(x_mat.shape[0]), int(len(target))

    start_total = time.perf_counter()

    a_full, lb_full, ub_full = _build_milp_constraints(n, k, target)
    integrality = np.ones(n * k, dtype=int)
    bounds = Bounds(lb=0.0, ub=1.0)

    p = np.ones(n, dtype=int)
    converged_iter = max_iter

    for iter_num in range(1, max_iter + 1):
        p_new, _ = _solve_milp_assignment(
            x_mat, centroids, target,
            a_full, lb_full, ub_full,
            integrality, bounds,
            dist_method,
            time_limit=time_limit,
        )

        new_centroids = centroids.copy()
        for j in range(1, k + 1):
            idx = np.where(p_new == j)[0]
            if len(idx) > 0:
                new_centroids[j - 1] = x_mat[idx].mean(axis=0)

        if float(np.max(np.abs(centroids - new_centroids))) < convergence_tol:
            converged_iter = iter_num
            p = p_new
            break

        centroids = new_centroids
        p = p_new

    total_time = time.perf_counter() - start_total

    if dist_matrix is None:
        dist_matrix = squareform(pdist(x_mat, metric=dist_method))

    metrics = compute_metrics(y_true, p, x_mat, dist_matrix=dist_matrix)
    card_pred = tabulate_clusters(p, k)

    return {
        "y_predict": p,
        "cardinality_pred": card_pred,
        "Execution_Time": float(total_time),
        "converged_iter": converged_iter,
        "ARI": metrics["ARI"],
        "AMI": metrics["AMI"],
        "NMI": metrics["NMI"],
        "Silhouette_mean": metrics["Silhouette_mean"],
    }