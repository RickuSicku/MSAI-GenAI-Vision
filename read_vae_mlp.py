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

class VAE_MLP_Regularized(nn.Module):
    def __init__(self, latent_dim=32, dropout=0.1):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(784, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, 512)
        self.ln4 = nn.LayerNorm(512)
        self.fc5 = nn.Linear(512, 784)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.ln2(self.fc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = F.relu(self.ln3(self.fc3(z)))
        h = self.dropout(h)
        h = F.relu(self.ln4(self.fc4(h)))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_dir = Path(__file__).parent / "models"
    model = VAE_MLP_Regularized(latent_dim=32).to(device)

    model_path = models_dir / "vae_mlp_regularized.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded VAE_MLP_Regularized")
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
    recon = recon.cpu().view(-1, 1, 28, 28)

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle("VAE MLP Regularized Reconstructions", fontsize=16)

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
    save_path = outputs_dir / "vae_mlp_reconstructions.png"
    plt.savefig(save_path)
    print(f"Saved reconstructions to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
