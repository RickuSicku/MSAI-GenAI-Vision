import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
from read_vae_cnn_deep import VAE_CNN_Deep

st.set_page_config(layout="wide", page_title="Latent Vector Arithmetic")

st.title("VAE Latent Vector Arithmetic")
st.write("Visualize vector additions and subtractions between the average latent representations of different Fashion-MNIST classes.")

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = VAE_CNN_Deep(latent_dim=256).to(device)
    models_dir = Path(__file__).parent / "models" / "QUEST_models"
    model_path = models_dir / "vae_cnn_deep_latent256_lr0.0001.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def get_average_latents():
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])
    # Load dataset
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Collect 100 samples per class to compute stable average latent vectors
    class_samples = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(class_samples[label]) < 100:
            class_samples[label].append(img)
        if all(len(v) >= 100 for v in class_samples.values()):
            break
            
    model = load_model()
    avg_latents = {}
    with torch.no_grad():
        for i in range(10):
            imgs = torch.stack(class_samples[i]).to(device)
            mu, _ = model.encode(imgs)
            # Average over the batch and move to CPU for caching
            avg_latents[i] = mu.mean(dim=0).cpu()
            
    return avg_latents

fashion_classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

model = load_model()
avg_latents = get_average_latents()

st.sidebar.header("Vector Operations")
base_idx = st.sidebar.selectbox("1. Base Class", range(10), format_func=lambda x: fashion_classes[x], index=9) # Default Ankle boot

options = ["None"] + fashion_classes

add_selection = st.sidebar.selectbox("2. Class to Add", options, index=options.index('Sandal'))
sub_selection = st.sidebar.selectbox("3. Class to Subtract", options, index=options.index('Sneaker'))

# Get indices if not None
add_idx = fashion_classes.index(add_selection) if add_selection != "None" else None
sub_idx = fashion_classes.index(sub_selection) if sub_selection != "None" else None

# Compute the resulting latent vector
z_result = avg_latents[base_idx].clone()
equation = fashion_classes[base_idx]

if add_idx is not None:
    z_result += avg_latents[add_idx]
    equation += f" + {fashion_classes[add_idx]}"

if sub_idx is not None:
    z_result -= avg_latents[sub_idx]
    equation += f" - {fashion_classes[sub_idx]}"

st.subheader(f"Equation: {equation}")

# Helper to decode and display an image
def decode_and_plot(z_tensor, title):
    with torch.no_grad():
        z_tensor = z_tensor.unsqueeze(0).to(device)
        img = model.decode(z_tensor).cpu().squeeze().numpy()
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    return fig

# Display columns
cols = st.columns(4)

with cols[0]:
    st.write("**Base**")
    st.pyplot(decode_and_plot(avg_latents[base_idx], fashion_classes[base_idx]))

if add_idx is not None:
    with cols[1]:
        st.write("**Add**")
        st.pyplot(decode_and_plot(avg_latents[add_idx], fashion_classes[add_idx]))

if sub_idx is not None:
    with cols[2]:
        st.write("**Subtract**")
        st.pyplot(decode_and_plot(avg_latents[sub_idx], fashion_classes[sub_idx]))

with cols[3]:
    st.write("**Result**")
    st.pyplot(decode_and_plot(z_result, "Generated Output"))