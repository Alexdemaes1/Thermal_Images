# ✅ Generador de imágenes térmicas sintéticas con DIFFUSION MODEL (DDPM)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
DATASET_PATH = "/Users/carlosbarroso/Downloads/Visual_Lab_V4/"
OUTPUT_PATH = "./diffusion_generated/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------- DATASET ---------------- #
class ThermalDataset(Dataset):
    def __init__(self, root_dir):
        self.txt_files = []
        for paciente in os.listdir(root_dir):
            path = os.path.join(root_dir, paciente)
            if not os.path.isdir(path): continue
            for f in os.listdir(path):
                if f.endswith(".txt") and "normalizado" in f and "recorte" not in f:
                    self.txt_files.append(os.path.join(path, f))
        print(f"✅ {len(self.txt_files)} archivos cargados para Diffusion")

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        matrix = np.loadtxt(self.txt_files[idx])  # ya normalizado
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        return tensor

# ---------------- UNet SIMPLIFICADO ---------------- #
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.middle = nn.Conv2d(128, 128, 3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t_emb):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x_mid = F.relu(self.middle(x2))
        x3 = F.relu(self.dec2(x_mid + x2))
        out = torch.tanh(self.dec1(x3 + x1))
        return out

# ---------------- DDPM SAMPLER ---------------- #
class Diffusion:
    def __init__(self, model, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def sample(self, shape, device):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.n_steps)):
            alpha_bar_t = self.alpha_bar[t].to(device)
            beta_t = self.beta[t].to(device)

            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x_t = x
            x = (1 / torch.sqrt(self.alpha[t])) * (
                x - ((1 - self.alpha[t]) / torch.sqrt(1 - alpha_bar_t)) * self.model(x_t, t)
            ) + torch.sqrt(beta_t) * z
        return x

# ---------------- ENTRENAMIENTO ---------------- #
def train_diffusion():
    dataset = ThermalDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleUNet().to(device)
    optimizador = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    n_steps = 1000
    diffusion = Diffusion(model, n_steps=n_steps)

    print("🚀 Entrenando modelo Diffusion...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for x in loader:
            x = x.to(device)
            t = torch.randint(0, n_steps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            alpha_bar = diffusion.alpha_bar[t].view(-1, 1, 1, 1).to(device)
            x_t = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
            pred = model(x_t, t)
            loss = loss_fn(pred, noise)

            optimizador.zero_grad()
            loss.backward()
            optimizador.step()
            total_loss += loss.item()

        print(f"📆 Epoch {epoch+1}/20 - Loss: {total_loss/len(loader):.4f}")

        # Guardar muestra
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample = diffusion.sample((1, 1, 480, 640), device).squeeze().cpu().numpy()
                np.savetxt(os.path.join(OUTPUT_PATH, f"generated_{epoch+1}_ddpm.txt"), sample, fmt="%.4f")
                plt.imsave(os.path.join(OUTPUT_PATH, f"generated_{epoch+1}_ddpm.png"), sample, cmap='inferno', dpi=300)

    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "diffusion_model.pth"))
    print("✅ Entrenamiento completado y modelo guardado")

    return model, diffusion

# ---------------- GENERACIÓN ---------------- #
def generate_samples(model, diffusion, n_samples=5):
    print("✨ Generando imágenes sintéticas...")
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            sample = diffusion.sample((1, 1, 480, 640), device).squeeze().cpu().numpy()
            txt_path = os.path.join(OUTPUT_PATH, f"generated_{i}_ddpm.txt")
            img_path = os.path.join(OUTPUT_PATH, f"generated_{i}_ddpm.png")
            np.savetxt(txt_path, sample, fmt='%.4f')
            plt.imsave(img_path, sample, cmap='inferno', dpi=300)
    print("✅ Imágenes sintéticas guardadas")

# ---------------- MAIN ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, diffusion = train_diffusion()
generate_samples(model, diffusion, n_samples=5)
