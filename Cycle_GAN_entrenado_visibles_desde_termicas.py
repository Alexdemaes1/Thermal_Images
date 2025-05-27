from tensorflow.keras.models import load_model
import numpy as np
import cv2

import os

if os.name == "posix":
    os.system("sudo renice -15 -p " + str(os.getpid()))

    
# Cargar el generador entrenado
generator = load_model("thermal_generator_sano.keras")

# Generar una imagen térmica sintética
num_images = 5  # Número de imágenes a generar
for i in range(num_images):
    noise = np.random.normal(0, 1, (1, 100))  # Vector de ruido
    generated_image = generator.predict(noise)

    # Convertir imagen de [-1,1] a [0,255]
    generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)

    # Guardar la imagen generada
    image_filename = f"thermal_generated_{i}.png"
    txt_filename = f"thermal_generated_{i}.txt"
    
    cv2.imwrite(image_filename, generated_image[0])
    print(f"Imagen generada guardada como '{image_filename}'")
    
    # Guardar la matriz en formato .txt
    np.savetxt(txt_filename, generated_image[0], fmt='%d')
    print(f"Matriz guardada como '{txt_filename}'")
