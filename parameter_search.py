## imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path
import wandb

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor() 
])

train_dataset = torchvision.datasets.FashionMNIST(
    root=str(DATA_DIR), 
    train=True,
    download=True, 
    transform=transform
)

# 3. Create the DataLoader
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=4
)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl

class VAE_CNN_Deep(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder: deeper stack
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),   # 32x32x32
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x16x16
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 128x8x8
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride=1, padding=1), # 128x8x8
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 8 * 8)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),   # 128x8x8
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # 64x16x16
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 32x32x32
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 128, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def train(model, train_loader, save_path, epochs, lr, beta):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, total_rec, total_kl = 0, 0, 0

        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()

        avg_loss = total_loss / len(train_loader)
        avg_rec = total_rec / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.2f} | Recon: {avg_rec:.2f} | KL: {avg_kl:.2f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "recon_loss": avg_rec,
            "kl_loss": avg_kl
        })
        
    full_path = MODELS_DIR / save_path
    torch.save(model.state_dict(), str(full_path))

def train_sweep():
    run = wandb.init()
    config = wandb.config
    
    model = VAE_CNN_Deep(latent_dim=config.latent_dim)
    
    train(
        model=model,
        train_loader=train_loader,
        save_path=f"vae_cnn_deep_latent{config.latent_dim}_lr{config.learning_rate}.pth",
        epochs=config.epochs,
        lr=config.learning_rate,
        beta=config.beta
    )

if __name__ == "__main__":
    sweep_config = {
        "method": "random",
        "metric": {"name": "recon_loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3
        },
        "parameters": {
            "learning_rate": {"values": [1e-3, 5e-4, 1e-4]},
            "latent_dim": {"values": [32, 64, 128]},
            "beta": {"values": [0.5, 1.0, 1.5]},
            "epochs": {"value": 15}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="vae-parameter-search"
    )
    
    wandb.agent(sweep_id, function=train_sweep, count=10)