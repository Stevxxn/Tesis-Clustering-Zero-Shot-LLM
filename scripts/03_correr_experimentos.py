"""
scripts/03_correr_experimentos.py
=================================
Orquestador experimental que compara:
    1. Baseline sin restricciones: K-Means++
    2. Algoritmos del repositorio (sin LLM): KM-MILP, SCK1, CSCLP, HCAKC, K-MedoidsSC
    3. Algoritmos guiados por LLM (Zero-Shot):
        - LLM-KM-MILP (Qwen 3.8 27B y Gemma 3 4B)
        - LLM-Greedy-KM (Qwen 3.8 27B y Gemma 3 4B)
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

# Asegurar que Python encuentre las carpetas raíz
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from algoritmos_base._common import compute_metrics, tabulate_clusters
from algoritmos_base.km_milp import clustering_with_size_constraints
from algoritmos_base.sck1_final import sck1_iterativo
from algoritmos_base.csclp import run_csclp_algo
from algoritmos_base.hcakc import hcakc_specified
from algoritmos_base.kmedoids import sc_medoids, pam_kaufman_rousseeuw
from algoritmos_llm.llm_km_milp import run_llm_km_milp
from algoritmos_llm.llm_greedy_km import run_llm_greedy_km

DATASETS_LIMPIOS = [
    "ag_news_500",
    "emotion_500",
    "multiclass_sentiment_500",
    "sst5_500",
    "yahoo_answers_500",
]

CARPETA_DATOS = ROOT_DIR / "datos_tesis"
CARPETA_RESULTADOS = ROOT_DIR / "resultados"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)


def registrar_fila(resultados, dataset_name, nivel, algoritmo, y_true, y_pred, x_mat, dist_mat, exec_time, target):
    metrics = compute_metrics(y_true, y_pred, x_mat, dist_matrix=dist_mat)
    card_pred = tabulate_clusters(y_pred, len(target))
    resultados.append({
        "Dataset": dataset_name,
        "Nivel": nivel,
        "Algoritmo": algoritmo,
        "N": len(y_pred),
        "K": len(target),
        "ARI": round(metrics["ARI"], 4),
        "AMI": round(metrics["AMI"], 4),
        "NMI": round(metrics["NMI"], 4),
        "Silhouette": round(metrics["Silhouette_mean"], 4),
        "Tiempo_s": round(exec_time, 4),
        "Cardinalidad_Objetivo": str(target.tolist()),
        "Cardinalidad_Obtenida": str(card_pred.tolist()),
    })
    print(f"  [✓] {algoritmo:<24} | ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | Sil: {metrics['Silhouette_mean']:.4f} | {exec_time:.2f}s")


def main():
    resultados = []

    for ds in DATASETS_LIMPIOS:
        print(f"\n============================================================")
        print(f"PROCESANDO DATASET: {ds}")
        print(f"============================================================")

        x_mat = np.load(CARPETA_DATOS / "embeddings" / f"embeddings_{ds}.npy")
        y_raw = np.load(CARPETA_DATOS / "labels" / f"labels_{ds}.npy")
        
        # Normalizar etiquetas reales a 1..K
        _, y_true_0 = np.unique(y_raw, return_inverse=True)
        y_true = y_true_0 + 1
        k = int(len(np.unique(y_true)))
        target = np.bincount(y_true)[1:]  # [500, 500, ...]

        # Precalcular matriz de distancia coseno una sola vez por dataset
        dist_mat = squareform(pdist(x_mat, metric="cosine"))

        # --- NIVEL 1: BASELINE SIN RESTRICCIONES ---
        t0 = time.perf_counter()
        km_pred = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x_mat) + 1
        registrar_fila(resultados, ds, "1_Baseline", "K-Means++ (Sin Restric.)", y_true, km_pred, x_mat, dist_mat, time.perf_counter() - t0, target)

        # --- NIVEL 2: ALGORITMOS DEL REPOSITORIO (SIN LLM) ---
        # 2.1 KM-MILP
        t0 = time.perf_counter()
        res_milp = clustering_with_size_constraints(x_mat, target, time_limit=60.0)
        registrar_fila(resultados, ds, "2_Base_Constrained", "KM-MILP", y_true, res_milp["p"], x_mat, dist_mat, time.perf_counter() - t0, target)

        # 2.2 SCK1
        t0 = time.perf_counter()
        rng = np.random.default_rng(123)
        init_c = x_mat[rng.choice(len(x_mat), size=k, replace=False)]
        res_sck1 = sck1_iterativo(x_mat, k, init_c, target)
        registrar_fila(resultados, ds, "2_Base_Constrained", "SCK1", y_true, res_sck1["assignments"], x_mat, dist_mat, time.perf_counter() - t0, target)

        # 2.3 K-MedoidsSC
        t0 = time.perf_counter()
        medoids = pam_kaufman_rousseeuw(dist_mat, k)
        res_kmed = sc_medoids(dist_mat, k, target, medoids)
        registrar_fila(resultados, ds, "2_Base_Constrained", "K-MedoidsSC", y_true, res_kmed["y_predict"], x_mat, dist_mat, time.perf_counter() - t0, target)

        # 2.4 HCAKC
        t0 = time.perf_counter()
        pred_hcakc = hcakc_specified(x_mat, k, target, distance_method="cosine")
        registrar_fila(resultados, ds, "2_Base_Constrained", "HCAKC", y_true, pred_hcakc, x_mat, dist_mat, time.perf_counter() - t0, target)

        # 2.5 CSCLP (Construye DataFrame mínimo requerido por run_csclp_algo)
        t0 = time.perf_counter()
        df_tmp = pd.DataFrame(x_mat)
        df_tmp["class"] = y_true
        res_csclp = run_csclp_algo(df_tmp, target.tolist(), dist_method="cosine")
        registrar_fila(resultados, ds, "2_Base_Constrained", "CSCLP", y_true, res_csclp["y_predict"], x_mat, dist_mat, time.perf_counter() - t0, target)

        # --- NIVEL 3: PROPUESTA GUIADA POR LLM (QWEN Y GEMMA) ---
        for llm_name in ["qwen", "gemma"]:
            proto_path = CARPETA_DATOS / "prototipos" / f"prototipos_{llm_name}_{ds}.npy"
            if not proto_path.exists():
                print(f"  [!] No se encontró {proto_path}, saltando {llm_name}...")
                continue

            prototipos = np.load(proto_path)

            # 3.1 LLM-Greedy-KM
            res_g = run_llm_greedy_km(x_mat, y_true, prototipos, target, dist_matrix=dist_mat)
            registrar_fila(resultados, ds, "3_Propuesta_LLM", f"LLM-Greedy-KM ({llm_name.upper()})", y_true, res_g["y_predict"], x_mat, dist_mat, res_g["Execution_Time"], target)

            # 3.2 LLM-KM-MILP
            res_m = run_llm_km_milp(x_mat, y_true, prototipos, target, dist_matrix=dist_mat, time_limit=60.0)
            registrar_fila(resultados, ds, "3_Propuesta_LLM", f"LLM-KM-MILP ({llm_name.upper()})", y_true, res_m["y_predict"], x_mat, dist_mat, res_m["Execution_Time"], target)

    df_final = pd.DataFrame(resultados)
    out_csv = CARPETA_RESULTADOS / "metricas_consolidadas.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"\n[★] EXPERIMENTOS FINALIZADOS. Resultados guardados en: {out_csv}")


if __name__ == "__main__":
    main()