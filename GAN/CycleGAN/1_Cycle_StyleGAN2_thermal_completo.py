import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Rutas
DATASET_PATH = "/Users/carlosbarroso/Downloads/Visual_Lab_V4/"
OUTPUT_PATH = "./stylegan2_txt_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------- Dataset ---------------- #
class ThermalTxtDataset(Dataset):
    def __init__(self, root_dir):
        self.txt_files = []
        for paciente in os.listdir(root_dir):
            carpeta = os.path.join(root_dir, paciente)
            if not os.path.isdir(carpeta): continue
            for f in os.listdir(carpeta):
                if f.endswith(".txt") and "normalizado" in f and "recorte" not in f:
                    self.txt_files.append(os.path.join(carpeta, f))
        print(f"✅ {len(self.txt_files)} archivos .txt cargados")

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        file_path = self.txt_files[idx]
        matrix = np.loadtxt(file_path)  # ya normalizado
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        return tensor


# ---------------- Estilo GAN mínimo ---------------- #
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 64, 4, 1, 0),  # (N,100,1,1) -> (N,64,4,4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # (N,32,8,8)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # (N,1,16,16)
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),   # <-- ¡clave!
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ---------------- Entrenamiento ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ThermalTxtDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
loss_fn = nn.BCELoss()
g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4)
d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

epochs = 50
for epoch in range(epochs):
    for real in loader:
        real = real.to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, 100, 1, 1).to(device)

        # --- Discriminador
        fake = generator(noise)
        d_real = discriminator(real)
        d_fake = discriminator(fake.detach())
        d_loss = loss_fn(d_real, torch.ones_like(d_real)) + loss_fn(d_fake, torch.zeros_like(d_fake))
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # --- Generador
        fake = generator(noise)
        g_fake = discriminator(fake)
        g_loss = loss_fn(g_fake, torch.ones_like(g_fake))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    print(f"🌡️ Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Guardar muestra
    if epoch % 10 == 0:
        out = fake[0, 0].detach().cpu().numpy()
        np.savetxt(f"{OUTPUT_PATH}/generated_{epoch}.txt", out, fmt="%.4f")
        plt.imsave(f"{OUTPUT_PATH}/generated_{epoch}.png", out, cmap='inferno')

# Guardar modelo
torch.save(generator.state_dict(), f"{OUTPUT_PATH}/generator_final.pth")
print("✅ Entrenamiento completado y modelo guardado.")
