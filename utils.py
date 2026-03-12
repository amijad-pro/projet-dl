import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import scipy.io
import numpy as np
import streamlit as st

@st.cache_resource
def get_mnist_loaders(batch_size=128):
    """
    Télécharge et prépare les DataLoaders pour MNIST.
    Normalise les pixels entre 0 et 1 (pour que la focntione de perte de reconstruction puisse bien fonctionner)
    """
    # Transformation : Conversion en tenseur PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Téléchargement des jeux de données
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Création des itérateurs (DataLoaders)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


@st.cache_resource
def get_frey_loader(batch_size=128):
    """
    Charge le dataset Frey Face depuis le fichier .mat.
    """
    # Chargement des données MATLAB
    data = scipy.io.loadmat('data/frey_rawface.mat')
    # Les visages sont stockés dans une matrice (20x28, 1965)
    # On les transpose et on les normalise entre 0 et 1
    faces = data['ff'].T.astype('float32') / 255.0
    
    # Conversion en tenseur PyTorch (N, 1, 28, 20)
    faces = torch.from_numpy(faces).view(-1, 1, 28, 20)
    

    # Création du DataLoader
    dataset = torch.utils.data.TensorDataset(faces)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader

"""This module contains useful functions for other modules.
"""

@st.cache_resource
def get_fashion_mnist_loaders(batch_size=128):
    """
    Download and prepare train/test DataLoaders for FashionMNIST.
    Pixels are normalized to [0, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def poly(x, order=3):
    """Evaluates the different powers of an input vector.

    The input vector is evaluated element-wise
    to the power 1, 2, ..., `order`. The resulting vectors
    are then concatenated and returned.

    Parameters
    ----------
    x: array_like
        The input vector, of shape `(n, 1)`.
    order: int
        The maximum order to which the powers of
        `x`are computed.

    Returns
    -------
    x_out: array_like
        The concatenation of all
        the powers of `x`, of shape `(n, order)`.

    """
    x_out = x
    for i in range(2, order + 1):
        x_out = np.concatenate((x_out, np.power(x, i)), axis=1)
    return x_out


def paths(hidden_layers=2, dropout_rate=0.0):
    """File paths for model weights and metrics from model parameters.

    The input vector is evaluated element-wise
    to the power 1, 2, ..., `order`. The resulting vectors
    are then concatenated and returned.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.

    Returns
    -------
    path_model: string
        The file path of the model weights.
    path_metrics: string
        The file path of the model metrics computed during training.

    """

    base_name = (
        "saved_models/fmnist_mlp_hidden="
        + str(hidden_layers)
        + "_dropout_rate="
        + str(dropout_rate)
    )
    path_weights = base_name + ".pth"
    path_metrics = base_name + "_metrics.csv"
    return path_weights, path_metrics

