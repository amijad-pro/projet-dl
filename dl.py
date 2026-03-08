"""Functions for the deep learning mode.

Notes
-----
Inspired from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.
"""

<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
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
        x = x.view(-1, 784) # On aplatit l'image (28x28 -> 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, alpha=1.0, beta=1.0):
    """
    Somme de la perte de reconstruction et de la KL divergence
    """
    # On récupère la taille dynamiquement (784 pour MNIST, 560 pour Frey)
    input_dim = recon_x.size(1)
    
    # Perte de reconstruction (BCE)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    
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
    
    for batch_idx, (data, _) in enumerate(train_loader):
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
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, alpha, beta).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss
=======
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import streamlit as st

from viz import training_curves
from utils import paths


def get_FashionMNIST_datasets(batch_size=64, only_loader=True):
    """Loads and returns the FashionMNIST dataset.

    The data is returned as DataLoader objects by default or,
    if specified by the user, also as datasets.

    Parameters
    ----------
    batch_size: int, default=64
        The batch size to use for the DataLoader objects.
    only_loader: bool, default=True
        If `True`, returns only the DataLoader objects.
        If `False`, also returns the datasets.

    Returns
    -------
    train_dataloader: torch.utils.data.DataLoader
        The DataLoader of the training data.
    test_dataloader: torch.utils.data.DataLoader
        The DataLoader of the test data.
    training_data: torchvision.datasets.mnist.FashionMNIST
        The training data, only returned if `only_loader==False`.
    test_data: torchvision.datasets.mnist.FashionMNIST
        The test data, only returned if `only_loader==False`.

    """
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    if only_loader:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, test_dataloader, training_data, test_data


# Define model
class FMNIST_MLP(nn.Module):
    """The MLP model we train on the FashionMNIST dataset.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.

    Attributes
    ----------
    flatten: nn.Flatten
        A flatten layer.
    linear_relu_stack: nn.Sequential
        A stack of liner layers with ReLU
        activations.
    metrics: pd.DataFrame
        The training metrics dataframe.
    """

    def __init__(self, hidden_layers=2, dropout_rate=0.0):

        super().__init__()
        self.flatten = nn.Flatten()
        list_hidden = []
        for _ in range(hidden_layers - 1):
            list_hidden.append(nn.Linear(512, 512))
            list_hidden.append(nn.ReLU())
            list_hidden.append(nn.Dropout(dropout_rate))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *list_hidden,
            nn.Linear(512, 10),
        )
        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    def forward(self, x):
        """The forward pass.

        Parameters
        ----------
        x: Tensor
            The input tensor, of shape `(batch_size, 1, 28, 28)`.

        Returns
        -------
        logits: Tensor
            The unnormalized logits, of shape `(batch_size, 10)`.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_metrics(self, df):
        """Sets the metrics dataframe.

        Used when a saved model is loaded,
        to also load its past training metrics dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            The training metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = df

    def update_metrics(self, series):
        """Updates the metrics dataframe after one epoch.

        Parameters
        ----------
        series: pd.Series
            The new row of metrics to add at the
            end of the metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = pd.concat([self.metrics, series.to_frame().T], ignore_index=True)


def train_step(dataloader, model, loss_fn, optimizer, device, mode=None):
    """The training step for one epoch.

    Arguments
    ---------
    dataloader: torch.utils.data.DataLoader
        The training DataLoader.
    model: nn.Module
        The model.
    loss_fn: nn.modules._Loss
        The loss function.
    optimizer: torch.optim.optimizer.Optimizer
        The optimizer.
    device: str
        The device to use, `"gpu"` or `"cpu"`.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    train_loss: float
        The averaged loss on all the batches,
        which will be added to the metrics dataframe.
    correct: float
        The accuracy of all the predictions on the epoch,
        which will be added to the metrics dataframe.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (mode == "script") & (batch % 100 == 0):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    return train_loss, correct


def test_step(dataloader, model, loss_fn, device, mode=None):
    """The evaluation step after one epoch.

    Arguments
    ---------
    dataloader: torch.utils.data.DataLoader
        The test DataLoader.
    model: nn.Module
        The model.
    loss_fn: nn.modules._Loss
        The loss function.
    device: str
        The device to use, `"gpu"` or `"cpu"`.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    test_loss: float
        The averaged loss on all the batches,
        which will be added to the metrics dataframe.
    correct: float
        The accuracy of all the predictions on all the batches,
        which will be added to the metrics dataframe.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if mode == "script":
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    return test_loss, correct


def get_and_train_model(
    train_dataloader,
    test_dataloader,
    hidden_layers=2,
    dropout_rate=0.0,
    epochs=5,
    mode=None,
):
    """Creates and trains a model on the given dataset.

    Doesn't train if saved weights are found for the given hyperparameters
    (except the number of epochs). The MLP architecture is displayed.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        The DataLoader of the training data.
    test_dataloader: torch.utils.data.DataLoader
        The DataLoader of the test data.
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.
    epochs: int, default=5
        The number of epochs used for training.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    model: FMNIST_MLP
        The model.
    """
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "script":
        print(f"Using {device} device")

    # Create the model
    model = FMNIST_MLP(hidden_layers, dropout_rate)
    path_weights, path_metrics = paths(hidden_layers, dropout_rate)

    # Load the weights if they already exist
    if os.path.exists(path_weights):
        if mode == "script":
            print("model already exists, let us just load it")
        elif mode == "st":
            st.write("Found a saved model with given config")
        model.load_state_dict(torch.load(path_weights))
        metrics = pd.read_csv(path_metrics, index_col=0)
        model.set_metrics(metrics)
    model = model.to(device)
    if mode == "script":
        print(model)
    elif mode == "st":
        st.text("Model architecture:")
        st.text(model)

    # Train the model and save the weights if they don't exist
    if not os.path.exists(path_weights):
        if mode == "script":
            print("No existing model found")
        elif mode == "st":
            st.write("Didn't find an existing model, training a new one")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for t in range(epochs):
            if mode == "script":
                print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_acc = train_step(
                train_dataloader, model, loss_fn, optimizer, device, mode
            )
            test_loss, test_acc = test_step(
                test_dataloader, model, loss_fn, device, mode
            )

            # Saved the metrics in the model.metrics dataframe
            new_row = pd.Series(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            model.update_metrics(new_row)
            if (mode == "st") & ((t + 1) % 10 == 0):
                st.text(
                    f"End of epoch {t+1}, Test Error:\n Accuracy: {(100 * new_row['test_acc']):>0.1f}%, Avg loss: {new_row['test_loss']:>8f}"
                )

        if mode == "script":
            print("Done!")

        # Save the weights and the metrics dataframe
        torch.save(model.state_dict(), path_weights)
        model.metrics.to_csv(path_metrics)
        if mode == "script":
            print("Saved PyTorch Model State to " + path_weights)
    if mode == "script":
        print(model.metrics)
    return model


if __name__ == "__main__":

    mode = "script"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    args = parser.parse_args()

    train_dataloader, test_dataloader = get_FashionMNIST_datasets(64)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = get_and_train_model(
        train_dataloader,
        test_dataloader,
        hidden_layers=args.hidden,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        mode=mode,
    )
    training_curves(model, mode)
>>>>>>> 266a5492f9f3d75528fb99e658271c452215da96
