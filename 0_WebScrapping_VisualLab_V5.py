from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import urllib.parse
import re
import requests
import csv
import concurrent.futures
import threading

BASE_URL = "https://visual.ic.uff.br/dmi/prontuario/"
TXT_BASE_URL = "https://visual.ic.uff.br/dmi/bancovl"
PROFILE_PATH = "/Users/carlosbarroso/Library/Application Support/Google/Chrome"

options = Options()
options.debugger_address = "localhost:9222"

print("🚀 Conectando a Chrome ya abierto...")
driver = webdriver.Chrome(options=options)
driver_lock = threading.Lock()
csv_lock = threading.Lock()

# Crear carpetas base
os.makedirs("Healthy", exist_ok=True)
os.makedirs("Sick", exist_ok=True)
os.makedirs("Unknown", exist_ok=True)

# Preparar CSV resumen
csv_rows = []
csv_headers = ["ID", "Diagnosis", "Age", "Exam", "Folder"]

# Leer todos los archivos Lista_pacientes_X.html desde raíz
html_files = sorted([f for f in os.listdir('.') if f.startswith("Lista_pacientes_") and f.endswith(".html")])
print(f"📂 Se encontraron {len(html_files)} archivos de pacientes para procesar.")

urls = []

# Extraer URLs desde todos los archivos
for file in html_files:
    print(f"📄 Procesando {file}...")
    with open(file, "r", encoding="utf-8") as f:
        html = f.read()
    matches = re.findall(r'href="(details\.php\?id=\d+)"', html)
    full_urls = [urllib.parse.urljoin(BASE_URL, href) for href in matches]
    urls.extend(full_urls)

urls = sorted(list(set(urls)))
print(f"🔗 Total de URLs únicas extraídas: {len(urls)}")

def procesar_paciente(url):
    with driver_lock:
        id_paciente = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id", ["XXX"])[0]
        print(f"📁 Accediendo a paciente {id_paciente} → {url}")
        driver.get(url)
        time.sleep(2)
        html = driver.page_source

    diagnosis_match = re.search(r'Diagnosis\*:\s*<span>(.*?)<', html, re.IGNORECASE)
    if diagnosis_match:
        diagnosis_raw = diagnosis_match.group(1).strip().lower()
        if "healthy" in diagnosis_raw:
            diagnosis = "healthy"
            grupo = "Healthy"
        elif "sick" in diagnosis_raw:
            diagnosis = "sick"
            grupo = "Sick"
        else:
            diagnosis = diagnosis_raw
            grupo = "Unknown"
    else:
        diagnosis = "unknown"
        grupo = "Unknown"

    carpeta = os.path.join(grupo, f"paciente_{id_paciente.zfill(3)}")
    os.makedirs(carpeta, exist_ok=True)

    ruta_html = os.path.join(carpeta, f"ID_{id_paciente.zfill(3)}.html")
    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✅ HTML guardado en {ruta_html} (grupo: {grupo})")

    age_match = re.search(r'(\d{1,3}) years old', html)
    exam_date_match = re.search(r'Visit 1: (.*?)<', html)
    age = age_match.group(1) if age_match else 'N/A'
    exam = exam_date_match.group(1).strip() if exam_date_match else 'N/A'

    with open(os.path.join(carpeta, "metadatos.txt"), "w") as meta:
        meta.write(f"ID: {id_paciente}\n")
        meta.write(f"Diagnosis: {diagnosis}\n")
        meta.write(f"Age: {age}\n")
        meta.write(f"Exam: {exam}\n")

    with csv_lock:
        csv_rows.append([id_paciente, diagnosis, age, exam, carpeta])

    print("   🔍 Buscando archivos .txt en el HTML...")
    txt_files = re.findall(r'T(\d{4})\.\d+\.\d+\.[SD]\.\d{4}-\d{2}-\d{2}\.\d{2}\.txt', html)
    full_txt = re.findall(r'(T\d{4}\.\d+\.\d+\.[SD]\.\d{4}-\d{2}-\d{2}\.\d{2}\.txt)', html)

    if not full_txt:
        print("   ⚠️ No se encontraron archivos .txt en el HTML.")
    else:
        for filename, folder in zip(full_txt, txt_files):
            url_txt = f"{TXT_BASE_URL}/{folder}/{filename}"
            destino = os.path.join(carpeta, filename)
            try:
                time.sleep(1)
                response = requests.get(url_txt, auth=("CBMhealth", "Soco98!health"), verify=False)
                if response.status_code == 200:
                    with open(destino, "wb") as f:
                        f.write(response.content)
                    print(f"      ✅ {filename} descargado correctamente")
                else:
                    print(f"      ❌ {filename} no disponible ({response.status_code})")
            except Exception as e:
                print(f"      ❌ Error al descargar {filename}: {e}")

# Ejecutar tareas en paralelo (por paciente)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(procesar_paciente, urls)

# Guardar CSV resumen
with open("resumen_pacientes.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)
    writer.writerows(csv_rows)

print("\n🏁 Proceso completado. Archivo 'resumen_pacientes.csv' generado.")
