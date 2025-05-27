import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# CONFIGURACIÓN
DATASET_PATH = "/Users/carlosbarroso/Downloads/Visual_Lab_V4/"
OUTPUT_PATH = "./generated_images_spade_fullres/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------- Dataset ---------------- #
class ThermalDataset(Dataset):
    def __init__(self, root_dir):
        self.txt_files = []
        for paciente in os.listdir(root_dir):
            path = os.path.join(root_dir, paciente)
            if not os.path.isdir(path): continue
            for f in os.listdir(path):
                if f.endswith(".txt") and "normalizado" in f and "recorte" not in f:
                    self.txt_files.append(os.path.join(path, f))
        print(f"✅ {len(self.txt_files)} archivos cargados para SPADE")

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        file_path = self.txt_files[idx]
        matrix = np.loadtxt(file_path)  # ya normalizado
        input_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # (1,H,W)

        # Máscara sintética centrada para SPADE
        H, W = matrix.shape
        mask = np.zeros((H, W), dtype=np.float32)
        h_start = H // 3
        h_end = h_start + H // 3
        w_start = W // 4
        w_end = w_start + W // 2
        mask[h_start:h_end, w_start:w_end] = 1.0
        seg_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1,H,W)

        return seg_tensor, input_tensor

# ---------------- SPADE Modules ---------------- #
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 64
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, 3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, 3, padding=1)

    def forward(self, x, segmap):
        norm = self.param_free_norm(x)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return norm * (1 + gamma) + beta

class SPADEGen(nn.Module):
    def __init__(self, input_nc=1, label_nc=1):
        super().__init__()
        self.fc = nn.Conv2d(label_nc, 64, 3, padding=1)
        self.spade1 = SPADE(64, label_nc)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)

        self.spade2 = SPADE(64, label_nc)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, segmap):
        x = self.fc(segmap)
        x = self.spade1(x, segmap)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.spade2(x, segmap)
        x = self.relu2(x)
        x = self.conv2(x)
        return self.tanh(x)

# ---------------- Entrenamiento ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ThermalDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

G = SPADEGen().to(device)
opt = torch.optim.Adam(G.parameters(), lr=2e-4)
loss_fn = nn.L1Loss()

epochs = 50
for epoch in range(epochs):
    G.train()
    total_loss = 0
    for i, (seg, real) in enumerate(loader):
        seg, real = seg.to(device), real.to(device)
        fake = G(seg)
        loss = loss_fn(fake, real)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"🌡️ Epoch {epoch}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    # Guardar una muestra visual
    if epoch % 10 == 0:
        example = fake[0, 0].detach().cpu().numpy()
        matrix_path = os.path.join(OUTPUT_PATH, f"spade_matrix_epoch_{epoch}.txt")
        image_path = os.path.join(OUTPUT_PATH, f"spade_image_epoch_{epoch}.png")
        np.savetxt(matrix_path, example)
        plt.imshow(example, cmap='inferno')
        plt.axis('off')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

# Guardar modelo
torch.save(G.state_dict(), os.path.join(OUTPUT_PATH, "spade_generator.pth"))
print("✅ SPADE finalizado. Modelo y muestras guardadas.")
