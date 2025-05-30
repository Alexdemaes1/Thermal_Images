import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import cv2

# Parámetros del modelo
IMG_HEIGHT = 120
IMG_WIDTH = 160
LATENT_DIM = 100  # Tamaño del vector de ruido

# Función para cargar imágenes térmicas desde archivos .txt
def load_thermal_data(txt_path, shape=(IMG_HEIGHT, IMG_WIDTH)):
    """Carga datos térmicos desde un archivo .txt y los convierte en una matriz normalizada."""
    thermal_matrix = np.loadtxt(txt_path)
    thermal_image = cv2.resize(thermal_matrix, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    thermal_image = (thermal_image - thermal_image.min()) / (thermal_image.max() - thermal_image.min())  # Normalización [0,1]
    return np.expand_dims(thermal_image, axis=-1)  # Añadir canal

# Cargar todas las imágenes térmicas de pacientes sanos
def load_images(base_path, shape=(IMG_HEIGHT, IMG_WIDTH)):
    """Carga matrices térmicas normalizadas para el entrenamiento de la GAN."""
    X_train = []
    for patient_folder in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_folder)
        if os.path.isdir(patient_path):
            for file in os.listdir(patient_path):
                if file.endswith(".txt"):
                    txt_path = os.path.join(patient_path, file)
                    thermal_image = load_thermal_data(txt_path, shape)
                    X_train.append(thermal_image)
    X_train = np.array(X_train)
    print(f"Tamaño final del conjunto de entrenamiento: {X_train.shape}")
    return X_train

# Crear el Generador
def build_generator():
    """Crea el generador de la GAN."""
    model = Sequential()
    model.add(Dense(15 * 20 * 128, activation="relu", input_dim=LATENT_DIM))  # Capa densa inicial
    model.add(Reshape((15, 20, 128)))  # Redimensionar a una forma inicial
    model.add(UpSampling2D())  # 30x40
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(UpSampling2D())  # 60x80
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(UpSampling2D())  # 120x160
    model.add(Conv2D(1, kernel_size=3, padding="same", activation='tanh'))  # Imagen final

    print("Generador creado exitosamente.")
    return model

# Crear el Discriminador
def build_discriminator():
    """Crea el discriminador de la GAN."""
    model = Sequential()
    model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print("Discriminador creado exitosamente.")
    return model

# Construir y compilar la GAN
def build_gan(generator, discriminator):
    """Compila la GAN uniendo el generador y el discriminador."""
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    
    z = Input(shape=(LATENT_DIM,))
    generated_image = generator(z)
    discriminator.trainable = False  # Congelar discriminador
    validity = discriminator(generated_image)
    
    gan = Model(z, validity)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    print("GAN creada exitosamente.")
    return gan

# Entrenamiento de la GAN
def train_gan(base_path, epochs=1000, batch_size=16):
    """Entrena la GAN usando las matrices térmicas normalizadas de pacientes sanos."""
    X_train = load_images(base_path)
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))  # Vector de ruido
        fake_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"{epoch}: [D loss: {d_loss[0]} | acc: {100 * d_loss[1]}] [G loss: {g_loss}]")
    
    generator.save("thermal_generator_sano.keras")
    print("Modelo del generador guardado exitosamente.")

# Ruta donde están las matrices térmicas normalizadas de pacientes sanos
base_path = "/Users/carlosbarroso/Downloads/Visual_Lab_V1/"  # Modifica con la ruta correcta

# Entrenar la GAN
train_gan(base_path, epochs=1000, batch_size=16)
