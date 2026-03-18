"""Utility functions for dataset loading and VAE visualizations.

This module centralizes:
- dataset loading helpers for MNIST, FashionMNIST, and Frey Faces,
- utility helpers for feature engineering and model checkpoint paths,
- Streamlit plotting helpers for reconstructions, losses, and samples,
- explanatory Streamlit sections used by the app.
"""

from pathlib import Path
import scipy.io
import streamlit as st
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from config import FASHION_MNIST_CLASS_NAMES, FREY_FACES_PATH, DATA_DIR


def _default_transform():
    """Return the default image transform used by torchvision datasets."""
    return transforms.Compose([transforms.ToTensor()])


@st.cache_resource
def _get_torchvision_loaders(dataset_cls, batch_size, root):
    """Create train and test loaders for a torchvision dataset.

    Parameters
    ----------
    dataset_cls:
        Torchvision dataset class, such as ``datasets.MNIST``.
    batch_size:
        Number of samples per batch.
    root:
        Directory where the dataset is stored or downloaded.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Train and test dataloaders.
    """
    transform = _default_transform()
    train_dataset = dataset_cls(
        root=str(root),
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = dataset_cls(
        root=str(root),
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


@st.cache_resource
def get_mnist_loaders(batch_size):
    """Return train and test dataloaders for MNIST."""
    return _get_torchvision_loaders(datasets.MNIST, batch_size=batch_size, root=DATA_DIR)


@st.cache_resource
def get_frey_loader(batch_size, data_path = FREY_FACES_PATH):
    """Load the Frey Faces dataset from a MATLAB ``.mat`` file.

    Parameters
    ----------
    batch_size:
        Number of samples per batch.
    data_path:
        Path to the ``frey_rawface.mat`` file.

    Returns
    -------
    DataLoader
        Dataloader containing face tensors shaped as ``(N, 1, 28, 20)``.

    Raises
    ------
    FileNotFoundError
        If the Frey Faces file cannot be found.
    KeyError
        If the expected ``ff`` key is missing from the MATLAB file.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Frey Faces dataset not found: {data_path}"
        )

    data = scipy.io.loadmat(data_path)
    if "ff" not in data:
        raise KeyError("Expected key 'ff' in Frey Faces MATLAB file.")

    faces = data["ff"].T.astype("float32") / 255.0
    face_tensor = torch.from_numpy(faces).view(-1, 1, 28, 20)
    dataset = TensorDataset(face_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@st.cache_resource
def get_fashion_mnist_loaders(batch_size):
    """Return train and test dataloaders for FashionMNIST."""
    return _get_torchvision_loaders(
        datasets.FashionMNIST,
        batch_size=batch_size, root=DATA_DIR
    )


def _dataset_configs():
    """Return normalized dataset configuration metadata."""
    return {
        "MNIST": {
            "loader_fn": get_mnist_loaders,
            "input_dim": 28 * 28,
            "image_shape": (28, 28),
            "has_labels": True,
            "num_classes": 10,
            "class_names": [str(index) for index in range(10)],
            "title": "MNIST",
            "description": "Handwritten digit dataset.",
        },
        "FashionMNIST": {
            "loader_fn": get_fashion_mnist_loaders,
            "input_dim": 28 * 28,
            "image_shape": (28, 28),
            "has_labels": True,
            "num_classes": 10,
            "class_names": FASHION_MNIST_CLASS_NAMES,
            "title": "FashionMNIST",
            "description": "Clothing image dataset.",
        },
        "Frey Faces": {
            "loader_fn": get_frey_loader,
            "input_dim": 28 * 20,
            "image_shape": (28, 20),
            "has_labels": False,
            "num_classes": None,
            "class_names": None,
            "title": "Frey Faces",
            "description": "Face dataset for unsupervised reconstruction.",
        },
    }


def load_dataset(dataset_name, batch_size):
    """Load a dataset and return standardized metadata for the app.

    Parameters
    ----------
    dataset_name:
        One of ``MNIST``, ``FashionMNIST``, or ``Frey Faces``.
    batch_size:
        Number of samples per batch.

    Returns
    -------
    dict[str, Any]
        Dictionary containing loaders and dataset metadata.

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    """
    dataset_configs = _dataset_configs()
    if dataset_name not in dataset_configs:
        valid_names = ", ".join(dataset_configs)
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Expected one of: {valid_names}."
        )

    config = dataset_configs[dataset_name].copy()
    loader_fn = config.pop("loader_fn")

    if dataset_name == "Frey Faces":
        train_loader = loader_fn(
            batch_size=batch_size,
            data_path=FREY_FACES_PATH
        )
        config["train_loader"] = train_loader
        config["test_loader"] = None
    else:
        train_loader, test_loader = loader_fn(batch_size=batch_size)
        config["train_loader"] = train_loader
        config["test_loader"] = test_loader

    return config