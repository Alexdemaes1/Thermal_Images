import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Dar prioridad a TensorFlow en la asignación de recursos
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
import cv2

def load_thermal_data(txt_path, shape=(128, 128)):
    """Carga los datos térmicos desde un archivo .txt y los convierte en una imagen normalizada."""
    thermal_matrix = np.loadtxt(txt_path)
    thermal_image = cv2.resize(thermal_matrix, shape, interpolation=cv2.INTER_AREA)  # Reducción con interpolación óptima
    thermal_image = (thermal_image - 19.9) / (33.07 - 19.9)  # Normalización basada en el rango conocido
    return thermal_image

def load_images(base_path, shape=(128, 128)):
    """Carga imágenes visibles y térmicas desde subcarpetas de pacientes en base_path."""
    X_real, X_thermal = [], []
    
    for patient_folder in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_folder)
        if os.path.isdir(patient_path):
            for img_file in os.listdir(patient_path):
                if img_file.endswith(".jpg"):
                    img_path = os.path.join(patient_path, img_file)
                    txt_file = img_file.replace(".jpg", ".txt")
                    txt_path = os.path.join(patient_path, txt_file)
                    
                    if os.path.exists(txt_path):
                        image = cv2.imread(img_path)
                        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
                        image = image / 127.5 - 1  # Normalización a [-1, 1]
                        
                        thermal_image = load_thermal_data(txt_path, shape)
                        thermal_image = np.expand_dims(thermal_image, axis=-1)  # Añadir canal
                        
                        X_real.append(image)
                        X_thermal.append(thermal_image)
    
    return np.array(X_real), np.array(X_thermal)

def build_generator(input_shape=(128, 128, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Entrada del generador
    model.add(Dense(128 * 16 * 16, activation="relu"))
    model.add(Reshape((16, 16, 128)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same", activation='tanh'))
    return model

def train_gan(base_path, epochs=1000, batch_size=16):
    X_real, X_thermal = load_images(base_path)
    generator = build_generator()
    
    for epoch in range(epochs):
        idx = np.random.randint(0, X_real.shape[0], batch_size)
        real_imgs, thermal_imgs = X_real[idx], X_thermal[idx]
        
        fake_imgs = generator.predict(thermal_imgs)
        
        if epoch % 200 == 0:
            print(f"{epoch}: Generando imagen de prueba...")
            generator.save("thermal_to_visual_generator.keras")
            test_image(generator)

def test_image(generator):
    """Genera una imagen de prueba con el modelo actual"""
    thermal_image = np.loadtxt(r"/Users/carlosbarroso/Downloads/Visual_Lab_V1/737745/T0009.1.1.D.2012-10-24.00.txt")
    thermal_image = cv2.resize(thermal_image, (128, 128), interpolation=cv2.INTER_AREA)
    thermal_image = (thermal_image - 19.9) / (33.07 - 19.9)  # Normalización ajustada
    thermal_image = np.expand_dims(thermal_image, axis=[0, -1])  # Añadir batch y canal
    
    generated_image = generator.predict(thermal_image)
    generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
    
    cv2.imwrite("imagen_generada.jpg", generated_image[0])
    print("Imagen generada guardada como 'imagen_generada.jpg'")

# Llamada a la función principal de entrenamiento
train_gan(r"/Users/carlosbarroso/Downloads/Visual_Lab_V1", epochs=1000, batch_size=16)
