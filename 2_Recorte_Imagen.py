import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Configuración general
carpetas_principales = ['Healthy', 'Sick', 'Unknown']
base_path = os.path.join(os.getcwd(), 'BBDD')

def detectar_mama(matriz):
    alto, ancho = matriz.shape
    y_ini = int(alto * 0.4)
    y_fin = int(alto * 0.95)
    roi = matriz[y_ini:y_fin, :]
    imagen_8bit = (roi * 255).astype(np.uint8)
    _, umbral = cv2.threshold(imagen_8bit, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    cierre = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(cierre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mejor_puntaje = -1
    mejor_bbox = None
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        y_global = y + y_ini
        if w < 40 or h < 40:
            continue
        region = matriz[y_global:y_global + h, x:x + w]
        if region.size == 0:
            continue
        cx = x + w // 2
        cy = y_global + h // 2
        temp_max = np.max(region)
        temp_med = np.mean(region)
        centro_region = region[h//3:h*2//3, w//3:w*2//3]
        temp_min = np.min(centro_region) if centro_region.size > 0 else 0
        depresion = temp_med - temp_min
        centrado = 1.0 - abs(cx - ancho // 2) / (ancho / 2)
        ubicacion_baja = (y_global + h) / alto
        proporcion = w / h
        redondez = 1.0 - abs(proporcion - 1)
        puntaje = temp_med * 2 + depresion * 2 + redondez + centrado + ubicacion_baja * 2
        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_bbox = (x, y_global, w, h)

    if mejor_bbox:
        x, y, w, h = mejor_bbox
        extra_abajo = int(h * 0.075)
        y0 = max(0, y)
        y1 = min(alto, y + h + extra_abajo)
        x0 = max(0, x - int(w * 0.15))
        x1 = min(ancho, x + w + int(w * 0.15))
        return matriz[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)

    return matriz, (0, 0, ancho, alto)

def procesar_directorio(base_path):
    for carpeta in carpetas_principales:
        carpeta_completa = os.path.join(base_path, carpeta)
        for paciente in os.listdir(carpeta_completa):
            ruta_paciente = os.path.join(carpeta_completa, paciente)
            if not os.path.isdir(ruta_paciente):
                continue

            for archivo in os.listdir(ruta_paciente):
                if archivo.endswith("_normalizado.txt"):
                    ruta_txt = os.path.join(ruta_paciente, archivo)
                    try:
                        matriz_norm = np.loadtxt(ruta_txt)
                        recorte, (x, y, w, h) = detectar_mama(matriz_norm)

                        base_name = archivo.replace("_normalizado.txt", "")
                        imagen_path = os.path.join(ruta_paciente, base_name + "_normalizado_recorte.jpeg")
                        txt_path = os.path.join(ruta_paciente, base_name + "_normalizado_recorte.txt")
                        rect_path = os.path.join(ruta_paciente, base_name + "_normalizado_recorte_rectangulo_previo.jpeg")

                        plt.imsave(imagen_path, recorte, cmap='inferno', vmin=0, vmax=1)
                        np.savetxt(txt_path, recorte, fmt='%.4f')

                        fig, ax = plt.subplots()
                        ax.imshow(matriz_norm, cmap='inferno', vmin=0, vmax=1)
                        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                        ax.add_patch(rect)
                        cx = x + w // 2
                        cy = y + h // 2
                        ax.plot(cx, cy, 'bo', markersize=6)
                        ax.axis('off')
                        fig.savefig(rect_path, bbox_inches='tight', pad_inches=0, dpi=300)
                        plt.close(fig)

                        print(f"Imagen recortada guardada: {imagen_path}")
                        print(f"Matriz recortada guardada: {txt_path}")
                        print(f"Imagen con rectángulo previa guardada: {rect_path}")

                    except Exception as e:
                        print(f"Error procesando {ruta_txt}: {e}")

if __name__ == "__main__":
    procesar_directorio(base_path)
