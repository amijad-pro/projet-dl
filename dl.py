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
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

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
        return torch.sigmoid(self.fc4(h3)) 

    def forward(self, x):
        x = x.reshape(x.size(0), -1) 
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # encoder takes [x, one_hot(y)]
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # decoder takes [z, one_hot(y)]
        self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def one_hot(self, y):
        return F.one_hot(y, num_classes=self.num_classes).float()

    def encode(self, x, y):
        y_onehot = self.one_hot(y)
        xy = torch.cat([x, y_onehot], dim=1)
        h1 = F.relu(self.fc1(xy))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_onehot = self.one_hot(y)
        zy = torch.cat([z, y_onehot], dim=1)
        h3 = F.relu(self.fc3(zy))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar



def _extract_data_and_labels(batch):
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
        labels = batch[1] if len(batch) > 1 and torch.is_tensor(batch[1]) else None

        if torch.is_tensor(data):
            return data, labels

        if isinstance(data, list):
            if len(data) == 1 and torch.is_tensor(data[0]):
                return data[0], labels
            return torch.stack(data), labels

    raise TypeError(f"Unsupported batch type: {type(batch)}")

def loss_function(recon_x, x, mu, logvar, alpha, beta):
    x = x.reshape(x.size(0), -1)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = alpha * BCE + beta * KLD
    return total_loss, BCE, KLD


def train_model(model, train_loader, optimizer, epoch, alpha=1.0, beta=1.0):
    model.train()
    train_loss = 0
    bce_loss = 0
    kld_loss = 0

    device = next(model.parameters()).device

    for batch_idx, batch in enumerate(train_loader):
        data, labels = _extract_data_and_labels(batch)
        data = data.to(device)
        labels = labels.to(device) if labels is not None else None

        optimizer.zero_grad()

        if labels is not None and hasattr(model, "num_classes"):
            recon_batch, mu, logvar = model(data, labels)
        else:
            recon_batch, mu, logvar = model(data)

        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, alpha, beta)
        loss.backward()

        train_loss += loss.item()
        bce_loss += bce.item()
        kld_loss += kld.item()

        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )

    n = len(train_loader.dataset)
    return {
        "total": train_loss / n,
        "bce": (alpha * bce_loss) / n,
        "kld": (beta * kld_loss) / n
    }


def test_model(model, test_loader, alpha=1.0, beta=1.0):
    model.eval()
    test_loss = 0
    bce_loss = 0
    kld_loss = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in test_loader:
            data, labels = _extract_data_and_labels(batch)
            data = data.to(device)
            labels = labels.to(device) if labels is not None else None

            if labels is not None and hasattr(model, "num_classes"):
                recon_batch, mu, logvar = model(data, labels)
            else:
                recon_batch, mu, logvar = model(data)

            loss, bce, kld = loss_function(recon_batch, data, mu, logvar, alpha, beta)
            test_loss += loss.item()
            bce_loss += bce.item()
            kld_loss += kld.item()

    n = len(test_loader.dataset)
    metrics = {
        "total": test_loss / n,
        "bce": (alpha * bce_loss) / n,
        "kld": (beta * kld_loss) / n,
    }
    return metrics