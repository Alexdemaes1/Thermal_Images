# sudo renice -20 -p 14549     
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt

# Definición de modelos CycleGAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# Configuración de rutas
DATASET_PATH = "/Users/carlosbarroso/Downloads/Visual_Lab_V1/"
OUTPUT_PATH = "./generated_images_V2/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Funciones para manejar datos térmicos
def load_thermal_matrix(file_path):
    return np.loadtxt(file_path, delimiter=None)  # Permitir cualquier tipo de separación

def normalize_matrix(matrix):
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

def save_thermal_image(thermal_matrix, output_path, shape=(480, 640)):
    thermal_image = cv2.resize(thermal_matrix, shape, interpolation=cv2.INTER_CUBIC)
    plt.imshow(thermal_image, cmap='inferno')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

class ThermalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.txt_files = []
        for patient_dir in self.patient_dirs:
            self.txt_files.extend([os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith(".txt")])
    
    def __len__(self):
        return len(self.txt_files)
    
    def __getitem__(self, idx):
        file_path = self.txt_files[idx]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        matrix = np.array([[float(num) for num in line.split()] for line in lines])
        matrix = normalize_matrix(matrix)
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # Agregar canal
        return matrix_tensor

# Cargar dataset
dataset = ThermalDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Inicialización de modelos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Definir optimizadores y pérdida
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Entrenamiento
epochs = 100
for epoch in range(epochs):
    for i, real_matrices in enumerate(dataloader):
        real_matrices = real_matrices.to(device)
        
        # Generación de imágenes falsas
        fake_matrices = generator(real_matrices)
        
        # Entrenar discriminador
        real_preds = discriminator(real_matrices)
        fake_preds = discriminator(fake_matrices.detach())
        
        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)
        
        d_loss = criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Entrenar generador
        fake_preds = discriminator(fake_matrices)
        g_loss = criterion(fake_preds, torch.ones_like(fake_preds))
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # Guardar imágenes generadas cada 10 épocas
    if epoch % 10 == 0:
        matrix_output_path = os.path.join(OUTPUT_PATH, f"generated_epoch_{epoch}_V2.txt")
        np.savetxt(matrix_output_path, fake_matrices.cpu().detach().numpy()[0, 0])
        image_output_path = os.path.join(OUTPUT_PATH, f"generated_epoch_{epoch}_V2.png")
        save_thermal_image(fake_matrices.cpu().detach().numpy()[0, 0], image_output_path)

# Guardar modelo
torch.save(generator.state_dict(), "generator_V2.pth")
print("Entrenamiento completado. Modelo guardado.")

# Generar imágenes sintéticas
generator.eval()
with torch.no_grad():
    for i in range(5):
        sample = next(iter(dataloader)).to(device)
        generated_matrix = generator(sample).cpu().detach().numpy()[0, 0]
        txt_output_path = os.path.join(OUTPUT_PATH, f"synthetic_{i}_V2.txt")
        np.savetxt(txt_output_path, generated_matrix)
        png_output_path = os.path.join(OUTPUT_PATH, f"synthetic_{i}_V2.png")
        save_thermal_image(generated_matrix, png_output_path)
print("Matrices térmicas sintéticas y sus imágenes generadas y guardadas.")
