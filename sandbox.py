import torch
from dl import VAE, loss_function
from utils import get_mnist_loaders

# 1. Tester le chargement
train_loader, _ = get_mnist_loaders(batch_size=32)
images, _ = next(iter(train_loader))
print(f"Batch d'images chargé : {images.shape}") # Doit afficher [32, 1, 28, 28]

# 2. Tester le modèle
model = VAE(input_dim=784, latent_dim=20)
recon_batch, mu, logvar = model(images)
print(f"Reconstruction : {recon_batch.shape}") # Doit être [32, 784]

# 3. Tester la perte (Loss)
loss = loss_function(recon_batch, images, mu, logvar, alpha=1.0, beta=1.0)
print(f"Loss initiale : {loss.item():.2f}")
