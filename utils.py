import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import scipy.io
import numpy as np


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
