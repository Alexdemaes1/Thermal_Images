import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt

# Paths
DATASET_PATH = "/Users/carlosbarroso/Downloads/Visual_Lab_V4/"
OUTPUT_PATH = "./generated_images_cut/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Dataset térmico
class ThermalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patient_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.txt_files = []
        for patient_dir in self.patient_dirs:
            self.txt_files.extend([
                os.path.join(patient_dir, f)
                for f in os.listdir(patient_dir)
                if f.endswith(".txt") and "normalizado" in f and "recorte" not in f # Normalizado sin recorte
            ])

        print(f"🔍 {len(self.txt_files)} archivos .txt encontrados para entrenamiento CUT")

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        file_path = self.txt_files[idx]
        with open(file_path, 'r') as f:
            lines = f.readlines()

        matrix = np.array([[float(num) for num in line.split()] for line in lines])
        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return tensor


# Generador CUT-like simple
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, 1, 3), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 7, 1, 3), nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Discriminador PatchGAN simple
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Utilidad para guardar imagen térmica
def save_thermal_image(matrix, path, shape=(480, 640)):
    resized = cv2.resize(matrix, shape, interpolation=cv2.INTER_CUBIC)
    plt.imshow(resized, cmap='inferno')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ThermalDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

# Contrastive loss (simplificado)
def contrastive_loss(fake, real):
    return torch.mean((fake - real) ** 2)

# Optimizers
g_opt = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCELoss()

# Entrenamiento
epochs = 50
for epoch in range(epochs):
    for i, real in enumerate(loader):
        real = real.to(device)
        fake = G(real)

        # Discriminador
        D_real = D(real)
        D_fake = D(fake.detach())
        loss_D = bce(D_real, torch.ones_like(D_real)) + bce(D_fake, torch.zeros_like(D_fake))
        d_opt.zero_grad()
        loss_D.backward()
        d_opt.step()

        # Generador con contraste
        D_fake = D(fake)
        loss_G = bce(D_fake, torch.ones_like(D_fake)) + contrastive_loss(fake, real)
        g_opt.zero_grad()
        loss_G.backward()
        g_opt.step()

        if i % 10 == 0:
            print(f"[{epoch}/{epochs}] Batch {i}: D_loss={loss_D.item():.3f}, G_loss={loss_G.item():.3f}")

    # Guardar muestras
    if epoch % 10 == 0:
        gen_matrix = fake[0, 0].detach().cpu().numpy()
        np.savetxt(os.path.join(OUTPUT_PATH, f"cut_matrix_epoch_{epoch}.txt"), gen_matrix)
        save_thermal_image(gen_matrix, os.path.join(OUTPUT_PATH, f"cut_image_epoch_{epoch}.png"))

# Guardar modelo
torch.save(G.state_dict(), os.path.join(OUTPUT_PATH, "generator_cut.pth"))
print("Modelo CUT guardado.")

