import torch
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from config import DEVICE


def plot_generated_samples(model, image_shape, latent_dim, n):
    """
    Generate random samples from the latent prior and plot them in a grid.

    Parameters
    ----------
    model : torch.nn.Module
        The trained VAE model used for decoding.
    image_shape : Tuple[int, int]
        The (height, width) of the images to be rendered.
    latent_dim : int
        The dimensionality of the latent space.
    n : int
        The number of samples to generate and display.
    """
    model.eval()
    latent_vectors = torch.randn(n, latent_dim, device=DEVICE)
    samples = _decode_samples(model, latent_vectors)
    plot_sample_grid(samples, image_shape)


def plot_conditional_samples(model, image_shape, latent_dim, class_idx, n):
    """
    Generate conditional samples for a specific class and plot them.

    Parameters
    ----------
    model : torch.nn.Module
        The trained CVAE model.
    image_shape : Tuple[int, int]
        The (height, width) of the output images.
    latent_dim : int
        The dimensionality of the latent space.
    class_idx : int
        The numerical index of the class to generate (e.g., 0 for 'T-shirt').
    n : int
        The number of samples to generate.
    """
    model.eval()
    latent_vectors = torch.randn(n, latent_dim, device=DEVICE)
    labels = torch.full((n,), class_idx, dtype=torch.long, device=DEVICE)
    samples = _decode_samples(model, latent_vectors, labels=labels)
    plot_sample_grid(samples, image_shape)


def plot_sample_grid(samples, image_shape):
    """
    Plot a horizontal grid of decoded image samples.

    Parameters
    ----------
    samples : torch.Tensor
        Tensors of decoded images with shape (num_samples, input_dim).
    image_shape : Tuple[int, int]
        The (height, width) for reshaping the flattened tensors.
    """
    num_samples = samples.shape[0]
    fig, axes = plt.subplots(1, num_samples, figsize=(2 * num_samples, 2.5))

    if num_samples == 1:
        axes = [axes]

    for index in range(num_samples):
        axes[index].imshow(
            samples[index].view(*image_shape).numpy(),
            cmap="gray",
        )
        axes[index].axis("off")

    _show_figure(fig)


def plot_latent_space(model, image_shape, latent_dim, n):
    """
    Display a 2D latent manifold by decoding a regular grid of latent coordinates.

    This visualization is only applicable if the latent dimension is at least 2.
    It iterates through a grid in the first two dimensions of the latent space.

    Parameters
    ----------
    model : torch.nn.Module
        The trained VAE model.
    image_shape : Tuple[int, int]
        The (height, width) of the images.
    latent_dim : int
        The total dimensionality of the latent space.
    n : int
        The number of images per side of the square grid (total n*n images).
    """
    if latent_dim < 2:
        st.warning("Latent dimension must be at least 2 to display the manifold.")
        return

    model.eval()
    height, width = image_shape
    figure = np.zeros((height * n, width * n))
    grid_x, grid_y = _grid_coordinates(n)

    for row_index, y_coord in enumerate(grid_y):
        for col_index, x_coord in enumerate(grid_x):
            latent_sample = torch.zeros(1, latent_dim, device=DEVICE)
            latent_sample[0, 0] = x_coord
            latent_sample[0, 1] = y_coord

            with torch.no_grad():
                decoded = model.decode(latent_sample)
                decoded_image = decoded.cpu().numpy().reshape(height, width)

            row_start = row_index * height
            row_end = (row_index + 1) * height
            col_start = col_index * width
            col_end = (col_index + 1) * width
            figure[row_start:row_end, col_start:col_end] = decoded_image

    fig, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(figure, cmap="gray")
    axis.axis("off")
    _show_figure(fig)


def plot_losses(history, test_history):
    """
    Plot total, reconstruction (BCE), and regularization (KL) losses across epochs.

    Parameters
    ----------
    history : dict
        Training history containing lists of losses for each metric.
    test_history : dict, optional
        Test history containing lists of losses for evaluation.
    """
    metric_specs = [
        ("Total", "Total Loss"),
        ("Reconstruction", "Reconstruction Loss"),
        ("Regularization (KL)", "KL Loss"),
    ]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(15, 4))
    epochs = list(range(1, len(history["Total"]) + 1))

    for axis, (metric_key, title) in zip(axes, metric_specs):
        _plot_series(
            axis,
            epochs,
            history,
            metric_key,
            title,
            test_history=test_history,
        )

    _show_figure(fig)


def plot_reconstructions(model, dataset, image_shape, input_dim, n):
    """
    Plot a comparison between original images and their VAE reconstructions.

    Parameters
    ----------
    model : torch.nn.Module
        The trained VAE/CVAE model.
    dataset : Any
        The dataset to sample original images from.
    image_shape : Tuple[int, int]
        The (height, width) for display.
    input_dim : int
        The flattened size of the input images.
    n : int
        The number of image pairs to display.
    """
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    with torch.no_grad():
        for index in range(n):
            image = get_image_from_dataset(dataset, index)
            model_input = image.view(1, input_dim).to(DEVICE)
            reconstruction, _, _ = model(model_input)

            axes[0, index].imshow(
                image.view(*image_shape).cpu().numpy(),
                cmap="gray",
            )
            axes[0, index].axis("off")
            axes[1, index].imshow(
                reconstruction.view(*image_shape).cpu().numpy(),
                cmap="gray",
            )
            axes[1, index].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstruction")
    _show_figure(fig)


def _decode_samples(model, latent_vectors, labels):
    """
    Internal utility to decode latent vectors, handling conditional logic.

    Parameters
    ----------
    model : torch.nn.Module
        The model with a .decode() method.
    latent_vectors : torch.Tensor
        Latent space coordinates.
    labels : torch.Tensor, optional
        Labels for conditional decoding.

    Returns
    -------
    torch.Tensor
        The decoded samples on CPU.
    """
    with torch.no_grad():
        if labels is None:
            return model.decode(latent_vectors).cpu()
        return model.decode(latent_vectors, labels).cpu()


def _grid_coordinates(n):
    """
    Generate coordinate arrays for a latent space grid.

    Parameters
    ----------
    n : int
        Number of points per axis.

    Returns
    -------
    grid_x : np.ndarray
        Coordinates for the x-axis.
    grid_y : np.ndarray
        Coordinates for the y-axis.
    """
    grid = np.linspace(-3, 3, n)
    return grid, grid


def _show_figure(fig):
    """
    Helper to render a Matplotlib figure in Streamlit and clear memory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.
    """
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_series(axis, epochs, train_history, key, title, test_history):
    """
    Internal helper to plot training and test curves for a specific metric.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The sub-plot axis to draw on.
    epochs : list
        The range of epoch numbers.
    train_history : dict
        Training metrics dictionary.
    key : str
        The dictionary key for the specific metric (e.g., 'total').
    title : str
        The title for the sub-plot.
    test_history : dict, optional
        Test metrics dictionary.
    """
    axis.plot(epochs, train_history[key], label="Train")
    if test_history is not None and test_history.get(key):
        axis.plot(epochs, test_history[key], label="Test")

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.legend()
    axis.grid(True)


def get_image_from_dataset(dataset, index):
    """
    Extract the image tensor from a dataset item, handling label presence.

    Parameters
    ----------
    dataset : Any
        The dataset object (e.g., torchvision.datasets.MNIST).
    index : int
        The index of the item to retrieve.

    Returns
    -------
    torch.Tensor
        The extracted image tensor.
    """
    item = dataset[index]
    return item[0] if isinstance(item, (tuple, list)) else item
