from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

# 1. Cargar una muestra de AG News (100 noticias para esta prueba rápida)
print("Cargando dataset...")
dataset = load_dataset("fancyzhx/ag_news", split="train")
textos_documentos = dataset['text'][:100]

# 2. Cargar el modelo Sentence-BERT
print("Cargando modelo de embeddings...")
modelo_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Generar embeddings de los documentos
print("Generando embeddings de documentos...")
embeddings_docs = modelo_sbert.encode(textos_documentos)

# 3. Simular la salida del LLM (Prototipos Semánticos Zero-Shot)
prototipos_llm = [
    "Noticias sobre política internacional, conflictos globales y relaciones diplomáticas.", # Mundo
    "Resultados de partidos, torneos atléticos, ligas profesionales y noticias deportivas.", # Deportes
    "Mercados financieros, empresas, acciones, economía y negocios corporativos.",           # Negocios
    "Avances tecnológicos, descubrimientos científicos, software, hardware y espacio."       # Ciencia/Tecnología
]

# Generar embeddings de los prototipos
print("Generando embeddings de los prototipos temáticos...")
embeddings_prototipos = modelo_sbert.encode(prototipos_llm)

# 4. Construir la Matriz de Costos
matriz_costos = cosine_distances(embeddings_docs, embeddings_prototipos)

print("\n--- Resultados ---")
print(f"Dimensiones de la matriz de costos: {matriz_costos.shape} (Documentos x Clústeres Temáticos)")
for i, costo in enumerate(matriz_costos[0]):
    print(f"Tema {i+1}: {costo:.4f}")