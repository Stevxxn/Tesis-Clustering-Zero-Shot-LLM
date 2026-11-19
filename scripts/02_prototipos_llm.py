"""
02_prototipos_llm.py
Módulo 2: Generación de prototipos semánticos Zero-Shot comparando:
    - Modelo 1 (Nube):  Qwen (qwen/qwen3.8-27b vía Groq)
    - Modelo 2 (Local): Gemma (gemma3:4b vía Ollama en localhost)
"""
import os
import re
import json
import time
import numpy as np
import torch
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONFIGURACIÓN DE MODELOS Y CONEXIONES
# ==============================================================================
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODELO_QWEN = "qwen/qwen3.8-27b"

# Configuración para Ollama Local (Cero costo, sin límites de cuota)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
MODELO_GEMMA = "gemma3:4b"

CARPETA_BASE = "datos_tesis"
NUM_DOCS_MUESTRA_LLM = 60
MAX_CHARS_POR_DOC = 300
SEMILLA = 42

DATASETS_NOMBRES = [
    "sst5_500",
    "emotion_500",
    "20_newsgroups_500",
    "multiclass_sentiment_500",
    "ag_news_500",
    "dbpedia_classes_500",
    "yelp_binary_500",
    "yahoo_answers_500"
]

def construir_prompt_zero_shot(textos_muestra, k):
    bloque_textos = "\n".join([f"- {t[:MAX_CHARS_POR_DOC].replace(chr(10), ' ')}" for t in textos_muestra])
    
    prompt = f"""You are an expert data scientist specializing in semantic text clustering.
Below is a random sample of {len(textos_muestra)} unlabeled documents from a text corpus.
Your task is to discover and define exactly {k} distinct, mutually exclusive latent themes/categories that best partition this entire corpus.

Rules:
1. Identify exactly {k} themes.
2. For each theme, write a rich, descriptive prototype paragraph (2 to 3 sentences) containing key concepts, vocabulary, and semantic context typical of documents belonging to that group.
3. Do NOT include numbering or bullet points inside the description text itself.
4. Respond ONLY with a valid JSON object using this exact schema:
{{
  "prototypes": [
    {{
      "cluster_id": 0,
      "theme_name": "Short label",
      "description": "Rich semantic description of 2-3 sentences..."
    }}
  ]
}}

Sample documents:
{bloque_textos}
"""
    return prompt

def parsear_prototipos(texto_resp, k):
    """Extrae y valida exactamente k descripciones, incluso si el JSON se truncó al final."""
    prototipos = []
    try:
        match = re.search(r"\{.*\}", texto_resp, re.DOTALL)
        contenido = json.loads(match.group(0) if match else texto_resp)
        prototipos = contenido.get("prototypes", [])
    except Exception:
        # Rescate elemento por elemento si el JSON quedó incompleto por límite de longitud
        objetos = re.findall(r'\{[^{}]*"description"\s*:\s*"[^"]*"[^{}]*\}', texto_resp, re.DOTALL)
        for obj_str in objetos:
            try:
                prototipos.append(json.loads(obj_str))
            except Exception:
                continue

    descripciones = [p["description"] for p in prototipos if "description" in p][:k]
    while len(descripciones) < k:
        descripciones.append(f"General topic cluster {len(descripciones)}")

    return descripciones, prototipos

def consultar_qwen_groq(client_groq, prompt, k, max_reintentos=3):
    """Consulta a Qwen en Groq respetando su límite de 1000 tokens de salida."""
    for intento in range(1, max_reintentos + 1):
        try:
            t0 = time.perf_counter()
            respuesta = client_groq.chat.completions.create(
                model=MODELO_QWEN,
                messages=[
                    {"role": "system", "content": "You output only valid JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=950,
                response_format={"type": "json_object"}
            )
            duracion = time.perf_counter() - t0
            texto_resp = respuesta.choices[0].message.content or ""
            descripciones, prototipos = parsear_prototipos(texto_resp, k)
            return descripciones, prototipos, duracion
        except Exception as e:
            if intento == max_reintentos:
                raise e
            espera = 3 * intento
            print(f"    [!] Qwen intento {intento} falló ({e}). Reintentando en {espera}s...")
            time.sleep(espera)

def consultar_gemma_local(client_ollama, prompt, k, max_reintentos=3):
    """Consulta a Gemma en Ollama usando el endpoint /v1/chat/completions con formato JSON nativo."""
    for intento in range(1, max_reintentos + 1):
        try:
            t0 = time.perf_counter()
            respuesta = client_ollama.chat.completions.create(
                model=MODELO_GEMMA,
                messages=[
                    {"role": "system", "content": "You output only valid JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                frequency_penalty=0.3,
                max_tokens=2000,
                extra_body={
                    "format": "json",
                    "options": {"num_ctx": 8192}
                }
            )
            duracion = time.perf_counter() - t0
            texto_resp = respuesta.choices[0].message.content or ""
            
            if not texto_resp.strip():
                raise ValueError("Respuesta vacía de Ollama")
                
            descripciones, prototipos = parsear_prototipos(texto_resp, k)
            return descripciones, prototipos, duracion
        except Exception as e:
            if intento == max_reintentos:
                raise e
            print(f"    [!] Gemma intento {intento} falló ({e}). Reintentando...")
            time.sleep(2)

def main():
    t_global_inicio = time.perf_counter()
    print("--- MÓDULO 2: PROTOTIPOS ZERO-SHOT (QWEN NUBE vs GEMMA LOCAL) ---")

    carpeta_prototipos = os.path.join(CARPETA_BASE, "prototipos")
    os.makedirs(carpeta_prototipos, exist_ok=True)

    client_groq = Groq(api_key=GROQ_API_KEY)
    client_ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    print(f"[*] Modelo 1 (Nube - Groq):    {MODELO_QWEN}")
    print(f"[*] Modelo 2 (Local - Ollama): {MODELO_GEMMA}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Cargando Sentence-BERT ('all-MiniLM-L6-v2') en {device.upper()}...")
    modelo_sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    rng = np.random.default_rng(SEMILLA)

    for nombre_ds in DATASETS_NOMBRES:
        txt_path = os.path.join(CARPETA_BASE, "textos", f"textos_{nombre_ds}.json")
        lbl_path = os.path.join(CARPETA_BASE, "labels", f"labels_{nombre_ds}.npy")

        if not (os.path.exists(txt_path) and os.path.exists(lbl_path)):
            print(f"\n[!] Saltando '{nombre_ds}': No se encontraron los archivos en '{CARPETA_BASE}/'.")
            continue

        print(f"\n==========================================")
        print(f"[*] Procesando prototipos para: {nombre_ds}")

        with open(txt_path, "r", encoding="utf-8") as f:
            textos = json.load(f)
        etiquetas = np.load(lbl_path)

        k = len(np.unique(etiquetas))
        print(f"[*] Número de clústeres objetivo (k): {k}")

        n_sample = min(NUM_DOCS_MUESTRA_LLM, len(textos))
        indices = rng.choice(len(textos), size=n_sample, replace=False)
        muestra_llm = [textos[i] for i in indices]

        prompt = construir_prompt_zero_shot(muestra_llm, k)

        # --- 1. QWEN (GROQ) ---
        out_qwen_npy = os.path.join(carpeta_prototipos, f"prototipos_qwen_{nombre_ds}.npy")
        out_qwen_json = os.path.join(carpeta_prototipos, f"prototipos_qwen_{nombre_ds}.json")

        if os.path.exists(out_qwen_npy) and os.path.exists(out_qwen_json):
            print("[➔] Prototipos QWEN YA EXISTEN. Omitiendo...")
        else:
            try:
                print(f"[*] Consultando QWEN ({MODELO_QWEN})...")
                desc_qwen, raw_qwen, t_qwen = consultar_qwen_groq(client_groq, prompt, k)
                emb_qwen = modelo_sbert.encode(desc_qwen, convert_to_numpy=True)
                
                np.save(out_qwen_npy, emb_qwen)
                with open(out_qwen_json, "w", encoding="utf-8") as f:
                    json.dump(raw_qwen, f, indent=2, ensure_ascii=False)
                print(f"[✓] QWEN listo en {t_qwen:.2f} s | Matriz: {emb_qwen.shape}")
            except Exception as e:
                print(f"[X] Error en QWEN para {nombre_ds}: {e}")

        # --- 2. GEMMA (OLLAMA LOCAL) ---
        out_gemma_npy = os.path.join(carpeta_prototipos, f"prototipos_gemma_{nombre_ds}.npy")
        out_gemma_json = os.path.join(carpeta_prototipos, f"prototipos_gemma_{nombre_ds}.json")

        if os.path.exists(out_gemma_npy) and os.path.exists(out_gemma_json):
            print("[➔] Prototipos GEMMA YA EXISTEN. Omitiendo...")
        else:
            try:
                print(f"[*] Consultando GEMMA LOCAL ({MODELO_GEMMA})...")
                desc_gemma, raw_gemma, t_gemma = consultar_gemma_local(client_ollama, prompt, k)
                emb_gemma = modelo_sbert.encode(desc_gemma, convert_to_numpy=True)
                
                np.save(out_gemma_npy, emb_gemma)
                with open(out_gemma_json, "w", encoding="utf-8") as f:
                    json.dump(raw_gemma, f, indent=2, ensure_ascii=False)
                print(f"[✓] GEMMA listo en {t_gemma:.2f} s | Matriz: {emb_gemma.shape}")
            except Exception as e:
                print(f"[X] Error en GEMMA para {nombre_ds}: {e}")

    t_global = time.perf_counter() - t_global_inicio
    print("\n==========================================")
    print(f"--- MÓDULO 2 FINALIZADO EN {t_global:.2f} SEGUNDOS ---")
    print("==========================================")

if __name__ == "__main__":
    main()