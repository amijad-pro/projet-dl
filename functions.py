from utils import get_mnist_loaders, get_frey_loader, get_fashion_mnist_loaders
import matplotlib.pyplot as plt
import torch
import streamlit as st
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(dataset_name: str, batch_size: int):
    if dataset_name == "MNIST":
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "input_dim": 28 * 28,
            "image_shape": (28, 28),
            "has_labels": True,
            "title": "MNIST",
            "description": "Handwritten digit dataset.",
        }

    if dataset_name == "FashionMNIST":
        train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)
        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "input_dim": 28 * 28,
            "image_shape": (28, 28),
            "has_labels": True,
            "title": "FashionMNIST",
            "description": "Clothing image dataset.",
        }

    if dataset_name == "Frey Faces":
        train_loader = get_frey_loader(batch_size=batch_size)
        return {
            "train_loader": train_loader,
            "test_loader": None,
            "input_dim": 28 * 20,
            "image_shape": (28, 20),
            "has_labels": False,
            "title": "Frey Faces",
            "description": "Face dataset for unsupervised reconstruction.",
        }

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_image_from_dataset(dataset, index: int):
    item = dataset[index]

    if isinstance(item, (tuple, list)):
        return item[0]
    return item


#  VISUALIZATION FUNCTIONS
def plot_reconstructions(model, dataset, image_shape, input_dim, n=10):
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    with torch.no_grad():
        for i in range(n):
            img = get_image_from_dataset(dataset, i)
            img_input = img.view(1, input_dim).to(device)

            recon, _, _ = model(img_input)

            axes[0, i].imshow(img.view(*image_shape).cpu().numpy(), cmap="gray")
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Original")

            axes[1, i].imshow(recon.view(*image_shape).cpu().numpy(), cmap="gray")
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Reconstruction")

    plt.tight_layout()
    st.pyplot(fig)


def plot_losses(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["Total"])
    axes[0].set_title("Total Loss")
    axes[0].grid(True)

    axes[1].plot(history["Reconstruction"])
    axes[1].set_title("Reconstruction Loss")
    axes[1].grid(True)

    axes[2].plot(history["Regularization (KL)"])
    axes[2].set_title("KL Loss")
    axes[2].grid(True)

    plt.tight_layout()
    st.pyplot(fig)


def plot_latent_space(model, image_shape, latent_dim, n=15):
    if latent_dim < 2:
        st.warning("Latent dimension must be at least 2 to display the manifold.")
        return

    model.eval()
    height, width = image_shape
    figure = np.zeros((height * n, width * n))

    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.zeros(1, latent_dim).to(device)
            z_sample[0, 0] = xi
            z_sample[0, 1] = yi

            with torch.no_grad():
                sample = model.decode(z_sample).cpu().numpy()
                digit = sample.reshape(height, width)
                figure[
                    i * height:(i + 1) * height,
                    j * width:(j + 1) * width,
                ] = digit

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(figure, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)


def plot_generated_samples(model, image_shape, latent_dim, n=8):
    model.eval()
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        samples = model.decode(z).cpu()

        for i in range(n):
            axes[i].imshow(samples[i].view(*image_shape).numpy(), cmap="gray")
            axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)