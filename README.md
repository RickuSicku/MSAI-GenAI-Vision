# Fashion-MNIST VAE Exploration & Latent Space Arithmetic

## Overview
This project explores **Variational Autoencoders (VAEs)** for generating and manipulating images from the Fashion-MNIST dataset. The work began with experimenting across multiple architectures, including **MLP-based VAEs** and **basic CNN VAEs**, leveraging the relatively small image size of Fashion-MNIST (`28×28`). Early experiments and comparisons are documented in `training_models.ipynb`.

After evaluating reconstruction quality and generated samples, the **deep CNN-based VAE** consistently produced sharper and more semantically meaningful outputs. As a result, subsequent efforts focused on refining this architecture through architectural improvements and large-scale hyperparameter optimization.

To systematically optimize performance, **Weights & Biases (W&B)** was integrated for experiment tracking and hyperparameter sweeps. Bayesian optimization sweeps were deployed on the **QUEST HPC cluster** using `submit_sweep.sh` and `parameter_search.py`, enabling efficient exploration of:
- Learning rates
- β values for β-VAE regularization
- Latent dimensionality
- Training configurations

Finally, to better understand and interact with the learned latent representations, an interactive **Streamlit dashboard** (`dashboard.py`) was developed. The dashboard computes stable class-level latent embeddings by averaging encoded vectors across multiple samples per class, enabling intuitive **latent space arithmetic** and real-time visualization of generated outputs.

---

# Model Architecture
### Deep CNN Variational Autoencoder (Latent Dimension = 64)

<img width="1408" height="768" alt="VAE Architecture" src="https://github.com/user-attachments/assets/fb81e4f8-1ec6-4784-84e0-3d6056c414dd" />

---
Note: The images are padded by two pixels before ingested by the model.

# Key Features & Additional Objectives

## 1. Hyperparameter Optimization on QUEST using W&B
Implemented a distributed hyperparameter tuning pipeline using **Weights & Biases Sweeps** on the **QUEST HPC cluster**. Bayesian optimization was used to efficiently search the parameter space and identify configurations that improved reconstruction quality and latent representation learning.

The sweep explored:
- Latent dimensions (e.g., 64, 128, 256)
- Learning rates
- β regularization coefficients
- Optimizer configurations

This setup enabled scalable experimentation while efficiently utilizing HPC compute resources. Public link to weights and biases dashboard on the hyperparameter sweep has been attached.

```bash
https://wandb.ai/ramakrishna1106s-northwestern-university/vae-parameter-search/sweeps/rm1o9v0y?nw=nwuserramakrishna1106s
```

---

## 2. Interactive Latent Space Arithmetic Dashboard
Developed a **Streamlit-based interactive dashboard** for real-time exploration of the learned latent space.

The application allows users to:
- Select Fashion-MNIST classes
- Perform latent vector arithmetic operations
- Decode the resulting latent representation into generated images instantly

### Example Operations
- `Ankle Boot + Sandal - Sneaker`
- `Coat + Shirt`
- `Dress - Pullover`

To improve semantic consistency, the dashboard computes **average latent vectors from 100 samples per class**, producing significantly cleaner and more stable arithmetic results compared to using single-image embeddings. Although storing trained model weights directly in a repository is generally discouraged due to size and portability concerns, the weights were included to ensure the dashboard remains easily runnable on local systems without requiring retraining.

---

# Challenges Faced & Solutions

## 1. Learning and Configuring QUEST HPC
This project involved first-time exposure to:
- SSH workflows
- HPC job scheduling
- Resource allocation
- Remote experiment management

A significant amount of time was spent understanding how to efficiently use cluster resources and debug incomplete or failed jobs. QUEST documentation, tutorials, and community resources were instrumental in overcoming these challenges.

---

## 2. Hyperparameter Optimization for Generative Models
Optimizing VAEs presented a unique challenge because the overall VAE loss combines:
- Reconstruction loss
- KL divergence regularization

A lower total loss does not always correspond to visually superior generations. To address this, hyperparameter sweeps prioritized minimizing **reconstruction loss**, which more directly correlates with perceptual image quality and preserved structural details. This resulted in more reliable model selection during experimentation.

---

## 3. Noisy Latent Arithmetic Outputs
Initial latent arithmetic experiments used embeddings from individual randomly selected images, leading to unstable and noisy generated outputs.

This issue was mitigated by:
1. Encoding multiple images per class
2. Averaging their latent vectors
3. Using the averaged embedding as a canonical class representation

This significantly improved the consistency and interpretability of latent arithmetic operations.

---

# Project Structure

```bash
├── training_models.ipynb      # Initial VAE experiments and architecture comparisons
├── read_vae_cnn_deep.py       # Generates an image for each of the classes and outputs them
├── parameter_search.py        # W&B hyperparameter sweep configuration
├── submit_sweep.sh            # QUEST HPC submission script
├── dashboard.py               # Streamlit latent arithmetic dashboard
├── models/QUEST_models        # Best model checkpoint.
└── README.md

outputs/ and QUEST_wandb_logs are present to verify usage of wandb and to showcase the generation of images. 
```

# Locally running the dashboard

```bash
pip install -r requirements.txt
python -m streamlit run dashboard.py
```


