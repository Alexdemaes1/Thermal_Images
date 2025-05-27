import os
import re
import csv
import unicodedata
from bs4 import BeautifulSoup

# Crear CSV de resumen completo
csv_rows = []
csv_headers = [
    "ID", "Diagnosis", "Age", "Exam", "Body Temperature", "Complaints", "Eating habits", "Further informations",
    "Mammography", "Radiotherapy", "Plastic surgery", "Prosthesis", "Biopsy", "Hormone replacement",
    "Nipple changes", "Wart signal", "Path",
    "Cancer family", "Family history", "Last menstrual period", "Menarche",
    "Signs", "Symptoms", "Two hours ago patient",
    "Smoked", "Drank coffee", "Consumed alcohol", "Physical exercise",
    "Applied products", "Marital status", "Race",
    "Diagnóstico", "Findings", "Descrição"
]

root_dirs = ["Healthy", "Sick", "Unknown"]

# Campos ya contemplados
campos_esperados = {
    "Complaints", "Eating habits", "Further informations",
    "Mammography", "Radiotherapy", "Plastic surgery", "Prosthesis",
    "Biopsy", "Use of hormone replacement", "Nipple changes",
    "Is there signal of wart on breast",
    "Cancer family", "Family history", "Last menstrual period", "Menarche",
    "Signs", "Symptoms", "Two hours ago patient",
    "Smoked", "Drank coffee", "Consumed alcohol", "Physical exercise",
    "Put some pomade, deodorant or products at breasts or armpits region",
    "Marital status", "Race",
    "Diagnóstico", "Findings", "Description", "Descrição"
}

def normalizar_campo(texto):
    texto = unicodedata.normalize('NFKD', texto)
    texto = texto.strip().rstrip("?:").lower()
    return texto

campos_esperados_normalizados = set(normalizar_campo(c) for c in campos_esperados)
campos_detectados = set()

for group in root_dirs:
    group_path = os.path.join(".", group)
    if not os.path.isdir(group_path):
        continue

    for paciente in os.listdir(group_path):
        paciente_dir = os.path.join(group_path, paciente)
        if not os.path.isdir(paciente_dir):
            continue

        html_file = os.path.join(paciente_dir, f"ID_{paciente[-3:]}.html")
        meta_out = os.path.join(paciente_dir, "metadatos_completo.txt")

        if not os.path.exists(html_file):
            print(f"❌ No se encontró HTML para {paciente}")
            continue

        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        content = soup.get_text(separator="\n")

        def extract_field(pattern):
            match = re.search(pattern, content, re.IGNORECASE)
            return match.group(1).strip() if match else ""

        # Campos básicos
        ID = extract_field(r'ID:\s*(\d+)') or paciente[-3:]
        diagnosis = extract_field(r'Diagnosis\*:\s*(Healthy|Sick)') or "unknown"
        age = extract_field(r'(\d{1,3}) years old')
        exam = extract_field(r'Visit 1:\s*(.*?)\n')
        temp = extract_field(r'Body temperature:\s*(\d{2}\.\d{2})')

        # Campos adicionales
        complaints = extract_field(r'Complaints:\s*(.*?)\n')
        eating = extract_field(r'Eating habits:\s*(.*?)\n')
        further = extract_field(r'Further informations[s]?:\s*(.*?)\n')
        mammo = extract_field(r'Mammography\?\s*(.*?)\n')
        radio = extract_field(r'Radiotherapy\?\s*(.*?)\n')
        surgery = extract_field(r'Plastic surgery\?\s*(.*?)\n')
        prosthesis = extract_field(r'Prosthesis\?:\s*(.*?)\n')
        biopsy = extract_field(r'Biopsy\?:\s*(.*?)\n')
        hormone = extract_field(r'Use of hormone replacement\?:\s*(.*?)\n')
        nipple = extract_field(r'Nipple changes\?:\s*(.*?)\n')
        wart = extract_field(r'Is there signal of wart on breast\?:\s*(.*?)\n')

        cancer_family = extract_field(r'Cancer family\?:\s*(.*?)\n')
        family_history = extract_field(r'Family history\?:\s*(.*?)\n')
        last_menstrual = extract_field(r'Last menstrual period:\s*(.*?)\n')
        menarche = extract_field(r'Menarche:\s*(.*?)\n')
        signs = extract_field(r'Signs:\s*(.*?)\n')
        symptoms = extract_field(r'Symptoms:\s*(.*?)\n')
        two_hours = extract_field(r'Two hours ago patient:\s*(.*?)\n')

        smoked = extract_field(r'Smoked\?\s*(.*?)\n')
        coffee = extract_field(r'Drank coffee\?\s*(.*?)\n')
        alcohol = extract_field(r'Consumed alcohol\?\s*(.*?)\n')
        exercise = extract_field(r'Physical exercise\?\s*(.*?)\n')
        applied = extract_field(r'Put some pomade.*?armpits region\?\s*(.*?)\n')

        marital = extract_field(r'Marital status:\s*(.*?)\.')
        race = extract_field(r'Race:\s*(.*?)\n')

        diagnostico = extract_field(r'Diagn[oó]stico\*?:\s*(.*?)\n')
        findings = extract_field(r'Findings:\s*(.*?)\n')
        descricao = extract_field(r'Descri[cç][aã]o:\s*(.*?)\n') or extract_field(r'Description:\s*(.*?)\n')

        # Guardar resumen CSV
        csv_rows.append([
            ID, diagnosis.lower(), age, exam, temp, complaints, eating, further,
            mammo, radio, surgery, prosthesis, biopsy, hormone, nipple, wart,
            paciente_dir, cancer_family, family_history, last_menstrual, menarche,
            signs, symptoms, two_hours, smoked, coffee, alcohol, exercise, applied,
            marital, race, diagnostico, findings, descricao
        ])

        # Guardar metadatos extendidos
        with open(meta_out, "w") as out:
            out.write(f"ID: {ID}\nDiagnosis: {diagnosis}\nAge: {age}\nExam: {exam}\nBody Temperature: {temp}\n")
            out.write(f"Complaints: {complaints}\nEating habits: {eating}\nFurther informations: {further}\n")
            out.write(f"Mammography: {mammo}\nRadiotherapy: {radio}\nPlastic surgery: {surgery}\n")
            out.write(f"Prosthesis: {prosthesis}\nBiopsy: {biopsy}\nUse of hormone replacement: {hormone}\n")
            out.write(f"Nipple changes: {nipple}\nWart signal: {wart}\n")
            out.write(f"Cancer family: {cancer_family}\nFamily history: {family_history}\n")
            out.write(f"Last menstrual period: {last_menstrual}\nMenarche: {menarche}\n")
            out.write(f"Signs: {signs}\nSymptoms: {symptoms}\nTwo hours ago patient: {two_hours}\n")
            out.write(f"Smoked: {smoked}\nDrank coffee: {coffee}\nConsumed alcohol: {alcohol}\n")
            out.write(f"Physical exercise: {exercise}\nApplied products: {applied}\n")
            out.write(f"Marital status: {marital}\nRace: {race}\n")
            out.write(f"Diagnóstico: {diagnostico}\nFindings: {findings}\nDescrição: {descricao}\n")

        # Guardar temperatura individual
        temp_file = os.path.join(paciente_dir, f"{ID.zfill(3)}_temperatura_body.txt")
        with open(temp_file, "w") as f:
            f.write(temp + "\n")

        # Detectar posibles nuevas variables
        for span in soup.find_all("span"):
            text = span.get_text(strip=True)
            campo_normalizado = normalizar_campo(text)
            if campo_normalizado and campo_normalizado not in campos_esperados_normalizados:
                campos_detectados.add(campo_normalizado)

# Escribir resumen global CSV
with open("resumen_completo.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(csv_rows)

# Guardar campos nuevos no contemplados
campos_nuevos = sorted(campos_detectados)
with open("variables_nuevas_detectadas.txt", "w", encoding="utf-8") as f:
    for campo in campos_nuevos:
        f.write(f"{campo}\n")

print("✅ Metadatos completos, resumen generado y nuevas variables detectadas guardadas.")
