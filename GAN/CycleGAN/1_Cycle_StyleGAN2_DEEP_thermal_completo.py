import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# CONFIG
DATASET_PATH = "/Users/carlosbarroso/Downloads/Visual_Lab_V4/"
OUTPUT_PATH = "./stylegan2_resultados_deep/"
GENERATOR_PATH = "./generator_deep.pth"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ----------- DEEPER GENERATOR -----------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ----------- DATASET -----------
class ThermalDataset(Dataset):
    def __init__(self, root_dir):
        self.txt_files = []
        for paciente in os.listdir(root_dir):
            path = os.path.join(root_dir, paciente)
            if not os.path.isdir(path): continue
            for f in os.listdir(path):
                if f.endswith(".txt") and "normalizado" in f and "recorte" not in f:
                    self.txt_files.append(os.path.join(path, f))
        print(f"✅ {len(self.txt_files)} archivos .txt normalizados cargados")

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        file_path = self.txt_files[idx]
        matrix = np.loadtxt(file_path)  # Ya está normalizado
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        return tensor, file_path

# ----------- GUARDADO -----------
def save_thermal(matrix, base_path):
    np.savetxt(base_path + ".txt", matrix, fmt="%.4f")
    plt.imsave(base_path + ".png", matrix, cmap='inferno', dpi=300)

# ----------- ENTRENAMIENTO -----------
def entrenar_y_guardar_modelo():
    model = Generator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.L1Loss()

    dataset = ThermalDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("🚀 Comienza entrenamiento...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch, _ in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📆 Epoch {epoch+1}/50 - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), GENERATOR_PATH)
    print(f"✅ Modelo guardado en {GENERATOR_PATH}")

# ----------- INFERENCIA EN BATCH -----------
def inferencia():
    model = Generator().to(device)
    model.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))
    model.eval()

    dataset = ThermalDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (input_tensor, path) in enumerate(loader):
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            output_np = output.squeeze().cpu().numpy()
            output_np = np.clip((output_np + 1) / 2, 0, 1)

            nombre = os.path.basename(path[0]).replace(".txt", "_deepgen")
            output_file = os.path.join(OUTPUT_PATH, nombre)
            save_thermal(output_np, output_file)

    print(f"✅ Inferencia completada para {len(dataset)} archivos.")

# ----------- MAIN -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(GENERATOR_PATH):
    entrenar_y_guardar_modelo()
else:
    print(f"🔁 Modelo encontrado: {GENERATOR_PATH}. Ejecutando inferencia...")
    inferencia()
