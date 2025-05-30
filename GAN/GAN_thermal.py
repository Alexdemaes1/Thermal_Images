import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, Activation, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import cv2

def load_thermal_data(txt_path, shape=(120, 160)):
    """Carga los datos térmicos desde un archivo .txt y los convierte en una matriz normalizada."""
    print(f"Cargando datos térmicos desde: {txt_path}")
    thermal_matrix = np.loadtxt(txt_path)
    print(f"Tamaño original de la matriz térmica: {thermal_matrix.shape}")
    
    # Redimensionar correctamente a (120, 160)
    thermal_image = cv2.resize(thermal_matrix, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    print(f"Tamaño después de resize: {thermal_image.shape}")
    
    # Normalización
    thermal_image = (thermal_image - np.min(thermal_image)) / (np.max(thermal_image) - np.min(thermal_image))
    return np.expand_dims(thermal_image, axis=-1)  # Añadir canal de profundidad

def load_images(base_path, shape=(120, 160)):
    """Carga matrices térmicas normalizadas para el entrenamiento de la GAN."""
    print(f"Cargando imágenes desde {base_path}")
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

def build_generator(input_shape=(120, 160, 1)):
    """Construye el generador de la GAN."""
    model = Sequential()
    model.add(Input(shape=(100,)))  # Vector de ruido
    model.add(Dense(120 * 160, activation="relu"))
    model.add(Reshape((120, 160, 1)))
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(UpSampling2D(size=(1, 1)))
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2D(1, kernel_size=3, padding="same", activation='tanh'))
    print("Generador creado exitosamente.")
    return model

def build_discriminator(input_shape=(120, 160, 1)):
    """Construye el discriminador de la GAN."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print("Discriminador creado exitosamente.")
    return model

def build_gan(generator, discriminator):
    """Compila la GAN uniendo el generador y el discriminador."""
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    
    z = Input(shape=(100,))  # Entrada de ruido
    generated_image = generator(z)
    discriminator.trainable = False  # Congelar discriminador
    validity = discriminator(generated_image)
    
    gan = Model(z, validity)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    print("GAN creada exitosamente.")
    return gan

def train_gan(base_path, epochs=1000, batch_size=16):
    """Entrena la GAN usando las matrices térmicas normalizadas."""
    X_train = load_images(base_path, shape=(120, 160))
    print(f"Tamaño de X_train después de carga: {X_train.shape}")
    
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))  # Vector de ruido
        fake_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"{epoch}: [D loss: {d_loss[0]} | acc: {100 * d_loss[1]}] [G loss: {g_loss}]")
    
    generator.save("thermal_to_visual_generator.keras")
    discriminator.save("thermal_to_visual_discriminator.keras")
    gan.save("thermal_to_visual_gan.keras")
    print("Modelos guardados exitosamente.")

# Ruta donde están las matrices térmicas normalizadas
base_path = os.getcwd()

# Entrenar la GAN
train_gan(base_path, epochs=1000, batch_size=16)
