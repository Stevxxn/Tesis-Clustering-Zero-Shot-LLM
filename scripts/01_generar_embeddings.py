"""
01_generar_embeddings.py
Generación de embeddings, etiquetas reales y textos de muestra (N por clase)
con medición de tiempo individual y tiempo total de ejecución.
"""
import os
import json
import time
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer

# ==============================================================================
# PARÁMETROS DE MUESTREO
# ==============================================================================
TIER_SELECCIONADO = "todos"  # "pequenos", "medianos", "gigantes" o "todos"
MUESTRAS_POR_CLASE = 500        # Cantidad exacta de documentos a tomar por clase
SEMILLA = 42                    # Garantiza reproducibilidad

DATASETS_POR_TAMANO = {
    "pequenos": [
        {"repo": "SetFit/sst5", "col_text": "text", "col_label": "label", "name": "sst5_500"},
        {"repo": "dair-ai/emotion", "col_text": "text", "col_label": "label", "name": "emotion_500"},
        {"repo": "SetFit/20_newsgroups", "col_text": "text", "col_label": "label", "name": "20_newsgroups_500"},
        {"repo": "Sp1786/multiclass-sentiment-analysis-dataset", "col_text": "text", "col_label": "label", "name": "multiclass_sentiment_500"}
    ],
    "medianos": [
        {"repo": "fancyzhx/ag_news", "col_text": "text", "col_label": "label", "name": "ag_news_500"},
        {"repo": "DeveloperOats/DBPedia_Classes", "col_text": "text", "col_label": "l1", "name": "dbpedia_classes_500"}
    ],
    "gigantes": [
        {"repo": "yassiracharki/Yelp_Reviews_for_Binary_Senti_Analysis", "col_text": "review_text", "col_label": "class_index", "name": "yelp_binary_500"},
        {"repo": "community-datasets/yahoo_answers_topics", "col_text": "question_title", "col_label": "topic", "name": "yahoo_answers_500"}
    ]
}

def obtener_muestra_balanceada(textos, etiquetas, muestras_por_clase=500, seed=42):
    """
    Extrae exactamente 'muestras_por_clase' documentos de cada clase.
    """
    rng = np.random.default_rng(seed)
    clases_unicas = np.unique(etiquetas)
    indices_seleccionados = []

    print(f"[*] Extrayendo {muestras_por_clase} instancias por clase:")
    for clase in clases_unicas:
        idx_clase = np.where(etiquetas == clase)[0]
        total_clase = len(idx_clase)
        
        n_muestra = min(total_clase, muestras_por_clase)
        idx_elegidos = rng.choice(idx_clase, size=n_muestra, replace=False)
        indices_seleccionados.extend(idx_elegidos)
        print(f"    - Clase {clase}: {n_muestra} seleccionadas (de {total_clase:,} totales)")

    indices_seleccionados = np.array(indices_seleccionados)
    rng.shuffle(indices_seleccionados)

    textos_muestra = [textos[i] for i in indices_seleccionados]
    etiquetas_muestra = etiquetas[indices_seleccionados]

    return textos_muestra, etiquetas_muestra

def main():
    # --- INICIO CRONÓMETRO GLOBAL ---
    tiempo_total_inicio = time.perf_counter()
    
    print(f"--- GENERACIÓN DE EMBEDDINGS ({MUESTRAS_POR_CLASE}/CLASE) | BLOQUE: {TIER_SELECCIONADO.upper()} ---")
    
    if TIER_SELECCIONADO == "todos":
        lista_datasets = (
            DATASETS_POR_TAMANO["pequenos"] + 
            DATASETS_POR_TAMANO["medianos"] + 
            DATASETS_POR_TAMANO["gigantes"]
        )
    elif TIER_SELECCIONADO in DATASETS_POR_TAMANO:
        lista_datasets = DATASETS_POR_TAMANO[TIER_SELECCIONADO]
    else:
        print(f"[X] Error: '{TIER_SELECCIONADO}' no es válido.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Dispositivo detectado: {device.upper()}\n")

    print("[*] Cargando modelo Sentence-BERT ('all-MiniLM-L6-v2')...")
    modelo_sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    for sub in ["embeddings", "labels", "textos"]:
        os.makedirs(os.path.join("datos_tesis", sub), exist_ok=True)

    for cfg in lista_datasets:
        repo = cfg["repo"]
        col_text = cfg["col_text"]
        col_label = cfg["col_label"]
        
        emb_path = os.path.join("datos_tesis", "embeddings", f"embeddings_{cfg['name']}.npy")
        lbl_path = os.path.join("datos_tesis", "labels", f"labels_{cfg['name']}.npy")
        txt_path = os.path.join("datos_tesis", "textos", f"textos_{cfg['name']}.json")

        print(f"\n==========================================")
        print(f"[*] Dataset: {repo}")

        if os.path.exists(emb_path) and os.path.exists(lbl_path) and os.path.exists(txt_path):
            print(f"[➔] Archivos YA EXISTEN para '{cfg['name']}'. Omitiendo...")
            continue

        try:
            ds_dict = load_dataset(repo)
            particiones = list(ds_dict.keys())
            
            splits_a_unir = [ds_dict[k] for k in particiones if k != 'unsupervised']
            ds_completo = concatenate_datasets(splits_a_unir)
            
            if col_text not in ds_completo.column_names:
                posibles = [c for c in ds_completo.column_names if c in ['text', 'sentence', 'review_text', 'content', 'question_title']]
                col_text = posibles[0] if posibles else col_text

            if col_label not in ds_completo.column_names:
                posibles_lbl = [c for c in ds_completo.column_names if c in ['label', 'labels', 'topic', 'l1', 'sentiment']]
                col_label = posibles_lbl[0] if posibles_lbl else col_label

            textos_completos = [t if t is not None else "" for t in ds_completo[col_text]]
            etiquetas_completas = np.array(ds_completo[col_label])

            textos_muestra, etiquetas_muestra = obtener_muestra_balanceada(
                textos_completos,
                etiquetas_completas,
                muestras_por_clase=MUESTRAS_POR_CLASE,
                seed=SEMILLA
            )

            print(f"[*] Total de instancias en la muestra final: {len(textos_muestra):,}")

            # Guardar etiquetas reales y textos crudos (para el LLM del Módulo 2)
            np.save(lbl_path, etiquetas_muestra)
            with open(txt_path, "w", encoding="utf-8") as f:
                json.dump(textos_muestra, f, ensure_ascii=False)

            # Generar y guardar embeddings
            t_inicio = time.perf_counter()
            embeddings = modelo_sbert.encode(
                textos_muestra,
                batch_size=512,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            t_fin = time.perf_counter()
            np.save(emb_path, embeddings)
            
            duracion = t_fin - t_inicio
            print(f"[✓] Guardado: {emb_path} | Forma: {embeddings.shape}")
            print(f"[✓] Tiempo codificación dataset: {duracion:.2f} s")

        except Exception as e:
            print(f"[X] Error al procesar {repo}: {e}")

    # --- FIN CRONÓMETRO GLOBAL ---
    tiempo_total_fin = time.perf_counter()
    tiempo_total = tiempo_total_fin - tiempo_total_inicio
    minutos = int(tiempo_total // 60)
    segundos = tiempo_total % 60

    print("\n==========================================")
    print(f"--- BLOQUE '{TIER_SELECCIONADO.upper()}' FINALIZADO ---")
    print(f"[⏱] TIEMPO TOTAL DE EJECUCIÓN DEL SCRIPT: {tiempo_total:.2f} segundos ({minutos} min {segundos:.2f} s)")
    print("==========================================")

if __name__ == "__main__":
    main()