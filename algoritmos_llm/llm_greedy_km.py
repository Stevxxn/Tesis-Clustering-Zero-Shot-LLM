"""
algoritmos_llm/llm_greedy_km.py
===============================
K-Means iterativo con restricción de tamaño mediante asignación voraz
ordenada (Sorted Greedy Assignment), inicializado con prototipos del LLM.
"""
from __future__ import annotations
import time
from typing import Any, Optional
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from algoritmos_base._common import compute_metrics, tabulate_clusters


def _greedy_constrained_assignment(
    cost_mat: np.ndarray, target_cardinality: np.ndarray
) -> np.ndarray:
    """
    Asigna cada punto al centroide más cercano respetando los cupos máximos.
    Ordena los puntos por su distancia mínima (los más seguros primero)[cite: 26].
    """
    n, k = cost_mat.shape
    remaining = target_cardinality.copy()
    labels = np.zeros(n, dtype=int)

    # Priorizar puntos con menor distancia a su centroide preferido
    min_dists = np.min(cost_mat, axis=1)
    point_order = np.argsort(min_dists)

    for i in point_order:
        # Ordenar centroides de más cercano a más lejano para el punto i[cite: 26]
        sorted_clusters = np.argsort(cost_mat[i])
        for c in sorted_clusters:
            if remaining[c] > 0:
                labels[i] = c + 1  # 1-based
                remaining[c] -= 1
                break

    return labels


def run_llm_greedy_km(
    x_mat: np.ndarray,
    y_true: np.ndarray,
    initial_centroids: np.ndarray,
    target_cardinality: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
    max_iter: int = 20,
    dist_method: str = "cosine",
    convergence_tol: float = 1e-4,
) -> dict[str, Any]:
    """
    Ejecuta K-Means con asignación voraz capacitada desde los prototipos del LLM.
    """
    x_mat = np.asarray(x_mat, dtype=float)
    centroids = np.asarray(initial_centroids, dtype=float).copy()
    target = np.asarray(target_cardinality, dtype=int)
    n, k = int(x_mat.shape[0]), int(len(target))

    start_total = time.perf_counter()
    p = np.ones(n, dtype=int)

    for _ in range(max_iter):
        cost_mat = cdist(x_mat, centroids, metric=dist_method)
        p_new = _greedy_constrained_assignment(cost_mat, target)

        new_centroids = centroids.copy()
        for j in range(1, k + 1):
            idx = np.where(p_new == j)[0]
            if len(idx) > 0:
                new_centroids[j - 1] = x_mat[idx].mean(axis=0)

        if float(np.max(np.abs(centroids - new_centroids))) < convergence_tol:
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
        "ARI": metrics["ARI"],
        "AMI": metrics["AMI"],
        "NMI": metrics["NMI"],
        "Silhouette_mean": metrics["Silhouette_mean"],
    }