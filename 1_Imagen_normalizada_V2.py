import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configuración general
carpetas_principales = ['Healthy', 'Sick', 'Unknown']
base_path = os.path.join(os.getcwd(), 'BBDD')
log_errores = os.path.join(base_path, 'errores_normalizacion_imagenes.txt')

# Iniciar archivo de errores
with open(log_errores, 'w') as f:
    f.write('Errores en archivos térmicos:\n')

def load_thermal_data(txt_path, shape=(480, 640)):
    """Carga los datos térmicos desde un archivo .txt y los convierte en una imagen normalizada, ignorando cabeceras no numéricas."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    numeric_lines = []
    for line in lines:
        try:
            float_values = [float(x) for x in line.strip().split()]
            numeric_lines.append(float_values)
        except ValueError:
            continue  # ignora líneas con texto

    thermal_matrix = np.array(numeric_lines)

    # Si la matriz es 1D, intenta convertirla a 2D (por ejemplo, reshape si es múltiplo de 640)
    if thermal_matrix.ndim == 1:
        if thermal_matrix.size % 640 == 0:
            thermal_matrix = thermal_matrix.reshape(-1, 640)
        else:
            raise ValueError(f"El archivo {txt_path} tiene una longitud inválida para convertir a matriz 2D")

    # thermal_image = cv2.resize(thermal_matrix, shape, interpolation=cv2.INTER_CUBIC)
    thermal_image = thermal_matrix  # conservar tamaño original
    thermal_image = (thermal_image - np.min(thermal_image)) / (np.max(thermal_image) - np.min(thermal_image))
    return thermal_image

def save_thermal_image(thermal_image, output_path_jpeg):
    """Guarda la imagen térmica como archivo JPEG en alta calidad."""
    plt.imshow(thermal_image, cmap='inferno')
    plt.axis('off')
    plt.savefig(output_path_jpeg, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# Recorremos todas las carpetas
for carpeta in carpetas_principales:
    carpeta_completa = os.path.join(base_path, carpeta)
    for paciente in os.listdir(carpeta_completa):
        ruta_paciente = os.path.join(carpeta_completa, paciente)
        if not os.path.isdir(ruta_paciente):
            continue

        # Buscar archivos .txt que empiecen por "T" y no contengan "normalizado"
        for archivo in os.listdir(ruta_paciente):
            if archivo.endswith('.txt') and archivo.startswith('T') and '_normalizado.txt' not in archivo:
                path_txt = os.path.join(ruta_paciente, archivo)

                try:
                    # Cargar y normalizar imagen térmica
                    thermal_image = load_thermal_data(path_txt)

                    # Guardar la imagen solo en JPEG
                    nombre_base = os.path.splitext(archivo)[0] + '_normalizado'
                    path_jpeg = os.path.join(ruta_paciente, nombre_base + '.jpeg')
                    save_thermal_image(thermal_image, path_jpeg)

                    # Guardar la matriz normalizada
                    path_txt_norm = os.path.join(ruta_paciente, nombre_base + '.txt')
                    np.savetxt(path_txt_norm, thermal_image, fmt='%.6f')

                    print(f"Imagen térmica guardada en: {path_jpeg}")
                    print(f"Matriz normalizada guardada en: {path_txt_norm}")

                except Exception as e:
                    with open(log_errores, 'a') as f:
                        f.write(f"{archivo}: {e}\n")
                    print(f"Error procesando {archivo}: {e}")
