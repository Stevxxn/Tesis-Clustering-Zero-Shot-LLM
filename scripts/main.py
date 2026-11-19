import subprocess
import sys

def ejecutar_script(nombre_archivo):
    print(f"\n---> Ejecutando {nombre_archivo} <---")
    subprocess.run([sys.executable, nombre_archivo], check=True)

def pipeline():
    scripts = [
        "scripts/01_generar_embeddings_muestra.py",
        "scripts/02_prototipos_llm.py",
    ]
    for script in scripts:
        ejecutar_script(script)

if __name__ == "__main__":
    pipeline()