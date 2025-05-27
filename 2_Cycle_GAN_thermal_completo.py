import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.utils import save_image

# Definición de modelos CycleGAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# Configuración de rutas
DATASET_PATH = "/Users/carlosbarroso/Downloads/Visual_Lab_V1/"
OUTPUT_PATH = "./generated_images/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Transformaciones para las imágenes
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ThermalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_files = []
        for patient_dir in self.patient_dirs:
            self.image_files.extend([os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith(".png")])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Cargar dataset
dataset = ThermalDataset(DATASET_PATH, transform=data_transform)
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
    for i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        
        # Generación de imágenes falsas
        fake_images = generator(real_images)
        
        # Entrenar discriminador
        real_preds = discriminator(real_images)
        fake_preds = discriminator(fake_images.detach())
        
        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)
        
        d_loss = criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Entrenar generador
        fake_preds = discriminator(fake_images)
        g_loss = criterion(fake_preds, torch.ones_like(fake_preds))
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # Guardar imágenes generadas cada 10 épocas
    if epoch % 10 == 0:
        save_image(fake_images, os.path.join(OUTPUT_PATH, f"generated_epoch_{epoch}.png"), normalize=True)

# Guardar modelo
torch.save(generator.state_dict(), "generator.pth")
print("Entrenamiento completado. Modelo guardado.")

# Generar imágenes sintéticas
generator.eval()
with torch.no_grad():
    for i in range(5):
        sample = next(iter(dataloader)).to(device)
        generated_image = generator(sample)
        save_image(generated_image, os.path.join(OUTPUT_PATH, f"synthetic_{i}.png"), normalize=True)
print("Imágenes sintéticas generadas y guardadas.")
