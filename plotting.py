import torch
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from config import DEVICE


def plot_generated_samples(model, image_shape, latent_dim, n):
    """Generate random samples from the latent prior and plot them."""
    model.eval()
    latent_vectors = torch.randn(n, latent_dim, device=DEVICE)
    samples = _decode_samples(model, latent_vectors)
    plot_sample_grid(samples, image_shape)


def plot_conditional_samples(model, image_shape, latent_dim, class_idx, n):
    """Generate conditional samples for a given class and plot them."""
    model.eval()
    latent_vectors = torch.randn(n, latent_dim, device=DEVICE)
    labels = torch.full((n,), class_idx, dtype=torch.long, device=DEVICE)
    samples = _decode_samples(model, latent_vectors, labels=labels)
    plot_sample_grid(samples, image_shape)


def plot_sample_grid(samples, image_shape):
    """Plot a horizontal grid of decoded samples."""
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
    """Display a 2D latent manifold by decoding a regular latent grid."""
    if latent_dim < 2:
        st.warning(
            "Latent dimension must be at least 2 to display the manifold."
        )
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
    """Plot total, reconstruction, and KL losses across epochs."""
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
    """Plot original images and their reconstructions."""
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
    """Decode latent vectors with or without conditional labels."""
    with torch.no_grad():
        if labels is None:
            return model.decode(latent_vectors).cpu()
        return model.decode(latent_vectors, labels).cpu()


def _grid_coordinates(n):
    """Return the default latent-space grid coordinates."""
    grid = np.linspace(-3, 3, n)
    return grid, grid


def _show_figure(fig):
    """Render a Matplotlib figure in Streamlit and release it."""
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_series(axis, epochs, train_history, key, title, test_history):
    """Plot a single training or test history series on one axis."""
    axis.plot(epochs, train_history[key], label="Train")
    if test_history is not None and test_history.get(key):
        axis.plot(epochs, test_history[key], label="Test")

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.legend()
    axis.grid(True)


def get_image_from_dataset(dataset, index):
    """Extract the image tensor from a dataset item.

    Supports datasets that return either a single tensor or a tuple like
    ``(image, label)``.
    """
    item = dataset[index]
    return item[0] if isinstance(item, (tuple, list)) else item
