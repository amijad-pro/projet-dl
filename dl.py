"""Functions for the deep learning mode.

Notes
-----
Inspired from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # ENCODEUR : compresse l'image en deux vecteurs, une moyenne (mu) et un écart-type (logvar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # DÉCODEUR : prend un point du latent space et essaie de reconstruire l'image
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick
        On ajoute un bruit aléatoire epsilon pour permettre la backpropagation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)) # Sigmoid pour avoir des pixels entre 0 et 1

    def forward(self, x):
        x = x.reshape(x.size(0), -1) # On aplatit l'image (28x28 -> 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def _extract_data(batch):
    """
    Extract input tensor from a batch.

    Handles:
    - data
    - (data,)
    - (data, label)
    - [data]
    - [data, label]
    """
    if torch.is_tensor(batch):
        return batch

    if isinstance(batch, (list, tuple)):
        data = batch[0]

        if torch.is_tensor(data):
            return data

        if isinstance(data, list):
            if len(data) == 1 and torch.is_tensor(data[0]):
                return data[0]
            return torch.stack(data)

    raise TypeError(f"Unsupported batch type: {type(batch)}")

def loss_function(recon_x, x, mu, logvar, alpha, beta):
    """
    Somme de la perte de reconstruction et de la KL divergence
    """
    # On récupère la taille dynamiquement (784 pour MNIST, 560 pour Frey)
    x = x.reshape(x.size(0), -1)
    
    # Perte de reconstruction (BCE)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # Perte de régularisation (KLD)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # On renvoie la perte totale ET les deux composantes séparées (visuel pour streamlit plus tard)
    total_loss = alpha * BCE + beta * KLD
    return total_loss, BCE, KLD


def train_model(model, train_loader, optimizer, epoch, alpha=1.0, beta=1.0):
    model.train() 
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    
    device = next(model.parameters()).device 
    
    for batch_idx, batch in enumerate(train_loader):
        data = _extract_data(batch).to(device)
        

        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Calcul de la perte avec nos paramètres alpha et beta (on récupère les 3 composantes)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, alpha, beta)
        
        # Backward pass 
        loss.backward()
        
        # Accumulation des pertes pour les statistiques
        train_loss += loss.item()
        bce_loss += bce.item()
        kld_loss += kld.item()
        
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    n = len(train_loader.dataset)
    # On renvoie un dictionnaire avec les moyennes par image
    return {
        "total": train_loss / n,
        "bce": (alpha * bce_loss) / n,
        "kld": (beta * kld_loss) / n
    }

def test_model(model, test_loader, alpha=1.0, beta=1.0):
    """
    Évalue le modèle sur les données de test.
    """
    model.eval() 
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1)
            recon_batch, mu, logvar = model(data)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar, alpha, beta)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss
