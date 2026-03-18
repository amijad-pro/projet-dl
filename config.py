import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_KEYS = ("Total", "Reconstruction", "Regularization (KL)")
DEFAULT_DATASET = "MNIST"
DATASET_OPTIONS = ["MNIST", "FashionMNIST", "Frey Faces"]

DATA_DIR = Path("./data")
FREY_FACES_PATH = DATA_DIR / "frey_rawface.mat"

FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
