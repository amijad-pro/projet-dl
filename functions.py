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
            "num_classes": 10,
            "class_names": [str(i) for i in range(10)],
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
            "num_classes": 10,
            "class_names": [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ],
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
            "num_classes": None,
            "class_names": None,
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


def plot_losses(history, test_history=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["Total"]) + 1)

    axes[0].plot(epochs, history["Total"], label="Train")
    if test_history is not None and test_history.get("Total"):
        axes[0].plot(epochs, test_history["Total"], label="Test")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["Reconstruction"], label="Train")
    if test_history is not None and test_history.get("Reconstruction"):
        axes[1].plot(epochs, test_history["Reconstruction"], label="Test")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history["Regularization (KL)"], label="Train")
    if test_history is not None and test_history.get("Regularization (KL)"):
        axes[2].plot(epochs, test_history["Regularization (KL)"], label="Test")
    axes[2].set_title("KL Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
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

def plot_sample_grid(samples, image_shape):
    n = samples.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i].imshow(samples[i].view(*image_shape).numpy(), cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

def plot_generated_samples(model, image_shape, latent_dim, n=8):
    model.eval()
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        samples = model.decode(z).cpu()

    plot_sample_grid(samples, image_shape)


def plot_conditional_samples(model, image_shape, latent_dim, class_idx, n=8):
    model.eval()
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        y = torch.full((n,), class_idx, dtype=torch.long, device=device)
        samples = model.decode(z, y).cpu()

    plot_sample_grid(samples, image_shape)



def render_model_explanations(dataset_info):
    st.markdown("""
    ### What this model is doing
    A **Variational Autoencoder (VAE)** learns how to compress an image into a small latent representation,
    then reconstruct it. The **encoder** maps an input image $x$ to the parameters of a latent Gaussian
    distribution, and the **decoder** reconstructs an image from a sampled latent vector $z$.

    The encoder predicts:
    - $\mu$: the mean of the latent distribution
    - $log(\sigma^2)$: the log-variance of the latent distribution

    A latent vector is then sampled and passed to the decoder. This is what lets the model both
    reconstruct inputs and generate brand-new samples.
    """)

    with st.expander("How the training parameters affect the model", expanded=True):
        st.markdown(r"""
        The VAE is trained with a weighted version of the usual ELBO objective:

        $$
        \mathcal{L}_{\text{total}} =
        \alpha \mathcal{L}_{\text{recon}} +
        \beta D_{KL}\big(q_\phi(z \mid x)\,\|\,p(z)\big)
        $$

        where:
        - Reconstruction loss measures how well the decoder rebuilds the input image.
        - KL divergence regularizes the latent distribution so it stays close to a standard normal prior.

        **Alpha ($\alpha$)** controls the weight of the reconstruction term.
        - Higher $\alpha$: sharper or more faithful reconstructions, but potentially a less organized latent space.
        - Lower $\alpha$: less pressure to perfectly reconstruct the training images.

        **Beta ($\beta$)** controls the weight of the KL term.
        - Higher $\beta$: stronger regularization, smoother and more structured latent space, but reconstructions may get blurrier.
        - Lower $\beta$: better reconstructions, but the latent space may become less smooth or less meaningful for generation.

        **Epochs** = the number of full passes through the training set.

        **Hidden dimension** = the width of the internal layers inside the encoder and decoder.
        A larger hidden dimension gives the model more capacity, but also increases training cost.

        Latent dimension = the size of the bottleneck vector $z$, the compressed representation learned by the model. 
        A larger latent dimension allows more information to be stored; smaller values force stronger compression. Minimum 2 enables 2D visualization of the latent space.
        """)

        

    if dataset_info["has_labels"]:
        st.info(
            "On labeled datasets such as MNIST and FashionMNIST, the app trains both a standard VAE and a CVAE. "
            "The standard VAE generates random digits or clothing items from the learned latent space. "
            "The Conditional VAE (CVAE) also uses class labels, so after training you can ask it to generate a specific digit or clothing category."
        )
    else:
        st.info(
            "For unlabeled datasets, the app trains a standard VAE. It learns a latent space that can reconstruct "
            "faces and generate new samples from that learned representation."
        )



def render_latent_space_explanation():
    st.markdown("""
    The **latent space** is the compressed bottleneck learned by the encoder.
    Each point in this space corresponds to a decoded image.

    **What to look for:**
    - nearby points should generate similar images
    - moving smoothly across the grid should gradually change the output
    - well-trained models often show structured transitions between styles, shapes, or classes

    In other words, this section helps you see whether the VAE learned a smooth and meaningful internal representation.
    """)