"""
01_generar_embeddings.py
Módulo 1: Generación y guardado de embeddings de documentos
"""
import os
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def main():
    print("--- INICIANDO FASE 1: PROCESAMIENTO DE DOCUMENTOS ---")
    
    nombre_dataset = "fancyzhx/ag_news"
    print(f"\n1. Cargando el dataset '{nombre_dataset}' desde Hugging Face...")
    dataset = load_dataset(nombre_dataset, split="train")
    
    TAMANO_MUESTRA = 100 
    print(f"   -> Extrayendo {TAMANO_MUESTRA} documentos...")
    textos_documentos = dataset['text'][:TAMANO_MUESTRA]
    
    nombre_modelo = 'all-MiniLM-L6-v2'
    print(f"\n2. Cargando el modelo Sentence-BERT ('{nombre_modelo}')...")
    modelo_sbert = SentenceTransformer(nombre_modelo)
    
    print("\n3. Calculando representaciones vectoriales (Embeddings)...")
    embeddings_docs = modelo_sbert.encode(textos_documentos, show_progress_bar=True)
    print(f"   -> Dimensiones resultantes: {embeddings_docs.shape} (Documentos x Dimensiones)")
    
    archivo_salida = 'embeddings_docs.npy'
    np.save(archivo_salida, embeddings_docs)
    
    ruta_completa = os.path.join(os.getcwd(), archivo_salida)
    print(f"   -> ¡Éxito! Archivo guardado en: {ruta_completa}")
    print("--- FASE 1 COMPLETADA ---")

if __name__ == "__main__":
    main()