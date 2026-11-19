import subprocess

def ejecutar_script(nombre_archivo):
    print(f"\n---> Ejecutando {nombre_archivo} <---")
    subprocess.run(["python", nombre_archivo], check=True)

def pipeline():
    scripts = [
        "scripts/01_generar_embeddings.py",
        "scripts/02_prototipos_llm.py",
        "scripts/03_matriz_costos.py",
        "scripts/04_asignacion_restringida.py"
    ]
    for script in scripts:
        ejecutar_script(script)

if __name__ == "__main__":
    pipeline()