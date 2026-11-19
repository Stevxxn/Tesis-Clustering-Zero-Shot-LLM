import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import json

# 1. Cargar los artefactos precalculados de la Fase 2
embeddings_prototipos = np.load('embeddings_prototipos.npy')
with open('nombres_temas.json', 'r') as f:
    nombres_temas = json.load(f)

# 2. Cargar el modelo (debe ser el mismo usado en el paso 1 y 2)
modelo_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Recibir el documento nuevo
nuevo_documento = "Apple announces a new revolutionary silicon chip with 16 cores for faster computing."

# 4. Procesar y comparar
embedding_nuevo = modelo_sbert.encode([nuevo_documento])
distancias = cosine_distances(embedding_nuevo, embeddings_prototipos)

# 5. Asignar al clúster con menor costo (menor distancia)
indice_ganador = np.argmin(distancias)
tema_asignado = nombres_temas[indice_ganador]
costo_asignacion = distancias[0][indice_ganador]

print(f"Documento nuevo: '{nuevo_documento}'")
print(f"Asignación Zero-Shot: Clúster '{tema_asignado}' (Costo: {costo_asignacion:.4f})")