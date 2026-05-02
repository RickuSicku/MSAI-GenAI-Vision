import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class VAE_CNN_Baseline(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder: 1x28x28 -> 64x7x7
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x7x7
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1x28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 64, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_dir = Path(__file__).parent / "models"
    model = VAE_CNN_Baseline(latent_dim=32).to(device)

    model_path = models_dir / "vae_cnn_baseline.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded VAE_CNN_Baseline")
    else:
        print(f"Could not find {model_path.name} in {models_dir}")
        return

    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    fashion_classes = (
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    )

    class_images = {}
    for image, label in dataset:
        if label not in class_images:
            class_images[label] = image
        if len(class_images) == 10:
            break

    images = torch.stack([class_images[i] for i in range(10)]).to(device)

    with torch.no_grad():
        recon, _, _ = model(images)

    images = images.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle("VAE CNN Baseline Reconstructions", fontsize=16)

    for i in range(10):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(fashion_classes[i])
        axes[0, i].axis('off')
        
        # Recon
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Reconstruction")

    plt.tight_layout()
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    save_path = outputs_dir / "vae_cnn_baseline_reconstructions.png"
    plt.savefig(save_path)
    print(f"Saved reconstructions to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
