# Fashion-MNIST VAE Exploration & Latent Space Arithmetic


## Overview
This project explores Variational Autoencoders (VAEs) on the Fashion-MNIST dataset. Initially, the project started by experimenting with different architectures, mainly MLP-based VAEs and basic CNN-based VAEs (as documented in `training_models.ipynb`). Based on the results of these preliminary experiments, the focus shifted to training a more robust **Deep CNN VAE**. 

To optimize the model's performance, **Weights & Biases (W&B)** was utilized to run a comprehensive hyperparameter sweep. This sweep was deployed on the **Quest Supercomputing Cluster** (`submit_sweep.sh` and `parameter_search.py`), allowing for an efficient search across different learning rates, beta values, and latent dimensions.

Finally, to intuitively visualize the learned representations, we developed an interactive **Streamlit dashboard** (`dashboard.py`). This tool computes the average latent vectors for each Fashion-MNIST class and allows users to perform interactive latent space arithmetic (e.g., creating novel outputs by adding or subtracting class representations).

## Extra Criteria Pursued
1. **Hyperparameter Tuning with W&B on a Supercomputer (Quest):** Successfully implemented and deployed a hyperparameter sweep using Weights & Biases on the Quest cluster to automatically search for the optimal latent dimensions (e.g., 256) and learning rates.
2. **Interactive Latent Arithmetic Dashboard (Streamlit):** Built a dynamic, interactive web application that enables real-time latent vector arithmetic. It allows users to naturally combine distinct clothing concepts (e.g., `Ankle Boot + Sandal - Sneaker`) and instantly decode the resulting 256-dimensional vector into an image.
3. **Deep Convolutional Architecture:** Scaled up the VAE from a simple baseline to a deep convolutional encoder-decoder architecture to improve feature extraction and reconstruction fidelity.

## Difficulties Faced & Solutions
1. **Hyperparameter Optimization Overhead:** Finding the right balance between reconstruction loss and KL divergence manually was time-consuming and computationally expensive. 
  * **Solution:** Automated the process by integrating `wandb.sweep` and writing a bash submission script (`submit_sweep.sh`) to utilize the university's Quest cluster for parallel, tracked training runs.
2. **Noisy Latent Arithmetic Results:** Initially, using the latent vector of a single random image for arithmetic operations resulted in highly variable and noisy generated images.
  * **Solution:** Improved the Streamlit dashboard logic to extract and average the latent vectors of 100 sample images per class. This created a highly stable, canonical representation for each category, yielding much cleaner arithmetic results.
---

### Setup & Running the Dashboard
To run the latent space exploration dashboard locally: (will take some time to load initially.)
```bash
# Ensure you have the required dependencies installed (torch, torchvision, streamlit, matplotlib)
python -m streamlit run dashboard.py
```
