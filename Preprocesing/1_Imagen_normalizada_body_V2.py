import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configuración
base_path = os.path.join(os.getcwd(), 'BBDD')
carpetas_principales = ['Healthy', 'Sick', 'Unknown']
salida_errores = os.path.join(base_path, 'errores_normalizacion_body.txt')

def load_body_temperature(txt_path):
    """Carga un valor de temperatura desde un archivo .txt."""
    with open(txt_path, 'r') as f:
        content = f.read().strip()
        if not content:
            raise ValueError("Archivo vacío")
        return float(content)

def normalize_relative_to_body(matriz, temp_body):
    """
    Normaliza una matriz térmica dividiendo todos los valores por la temperatura corporal del paciente,
    y luego reescala al rango [0, 1]. Esto asegura contraste incluso si los valores de la imagen
    son inferiores a la temperatura corporal, como ocurre habitualmente en termografía.
    """
    matriz_dividida = matriz / temp_body  # normalización relativa pura
    min_val = np.min(matriz_dividida)
    max_val = np.max(matriz_dividida)
    matriz_norm = (matriz_dividida - min_val) / (max_val - min_val)
    return np.clip(matriz_norm, 0, 1)

def save_body_image(value_norm, output_path):
    """Genera una imagen pequeña representando el valor de temperatura normalizado."""
    image = np.full((100, 300), value_norm)
    plt.imshow(image, cmap='inferno', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# Inicializa archivo de errores
with open(salida_errores, 'w') as err_file:
    err_file.write("Pacientes con errores en la normalización de temperatura del cuerpo:\n")

# Procesamiento
for carpeta in carpetas_principales:
    carpeta_completa = os.path.join(base_path, carpeta)
    for paciente in os.listdir(carpeta_completa):
        ruta_paciente = os.path.join(carpeta_completa, paciente)
        if not os.path.isdir(ruta_paciente):
            continue

        temp_body_value = None

        # Cargar la temperatura corporal si existe
        for archivo in os.listdir(ruta_paciente):
            if archivo.endswith('_temperatura_body.txt'):
                path_txt = os.path.join(ruta_paciente, archivo)
                try:
                    temp_body_value = load_body_temperature(path_txt)
                    temp_norm = (temp_body_value - 30.0) / 10.0  # Normalización absoluta para representación
                    nombre_base = archivo.replace('_temperatura_body.txt', '_body_normalizado')

                    # Guardar valor normalizado como .txt
                    path_txt_norm = os.path.join(ruta_paciente, nombre_base + '.txt')
                    np.savetxt(path_txt_norm, [temp_norm], fmt='%.6f')

                    # Guardar imagen .jpeg del valor corporal
                    path_img = os.path.join(ruta_paciente, nombre_base + '.jpeg')
                    save_body_image(temp_norm, path_img)

                    print(f"Temperatura normalizada guardada en: {path_txt_norm}")
                    print(f"Imagen JPEG guardada en: {path_img}")

                except ValueError as e:
                    with open(salida_errores, 'a') as err_file:
                        err_file.write(f"{archivo} - {str(e)}\n")
                    print(f"Error con {archivo}: {str(e)}")

        # Aplicar normalización relativa a todas las imágenes si hay temperatura corporal válida
        if temp_body_value is not None:
            for archivo in os.listdir(ruta_paciente):
                if archivo.endswith(".txt") and archivo.startswith("T") and "normalizado" not in archivo and "temperatura_body" not in archivo:
                    try:
                        matriz = np.loadtxt(os.path.join(ruta_paciente, archivo))
                        matriz_norm = normalize_relative_to_body(matriz, temp_body_value)

                        # Guardar como imagen y txt con sufijo _body_normalizado
                        nombre_base = os.path.splitext(archivo)[0] + '_body_normalizado'
                        path_img = os.path.join(ruta_paciente, nombre_base + '.jpeg')
                        path_txt = os.path.join(ruta_paciente, nombre_base + '.txt')

                        plt.imsave(path_img, matriz_norm, cmap='inferno', vmin=0, vmax=1)
                        np.savetxt(path_txt, matriz_norm, fmt='%.6f')

                        print(f"Imagen normalizada con temperatura corporal guardada en: {path_img}")
                        print(f"Matriz normalizada guardada en: {path_txt}")
                    except Exception as e:
                        print(f"Error procesando {archivo} para normalización con temp_body: {e}")
