"""Variational autoencoder models and training utilities.


-----
Inspired from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import streamlit as st

from config import DEVICE
from state import empty_loss_history, update_history


def reparameterize(mu, logvar):
    """Sample latent vector using the reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    """Basic variational autoencoder."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """Set up encoder and decoder layers."""
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encodeur
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Encode input into latent mean and log-variance."""
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CVAE(nn.Module):
    """Conditional variational autoencoder."""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        """Set up encoder and decoder layers with label conditioning."""
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def one_hot(self, y):
        """Convert labels to one-hot vectors."""
        return F.one_hot(y, num_classes=self.num_classes).float()

    def encode(self, x, y):
        y_onehot = self.one_hot(y)
        xy = torch.cat([x, y_onehot], dim=1)
        h = F.relu(self.fc1(xy))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def decode(self, z, y):
        y_onehot = self.one_hot(y)
        zy = torch.cat([z, y_onehot], dim=1)
        h = F.relu(self.fc3(zy))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, y):
        x = x.reshape(x.size(0), -1)
        mu, logvar = self.encode(x, y)
        z = reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


def unpack_batch(batch):
    """Get data and optional labels from a dataloader batch."""
    if torch.is_tensor(batch):
        return batch, None

    if isinstance(batch, (list, tuple)):
        data = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        return data, labels

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def loss_function(recon_x, x, mu, logvar, alpha, beta):
    """Compute reconstruction loss and KL divergence."""
    x = x.view(x.size(0), -1)
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = alpha * bce + beta * kld
    return total_loss, bce, kld


def run_model(model, data, labels=None):
    """Call the right forward signature depending on the model type."""
    if isinstance(model, CVAE):
        if labels is None:
            raise ValueError("CVAE requires labels, but labels=None was provided.")
        return model(data, labels)

    return model(data)

def train_model(model, train_loader, optimizer, epoch, alpha=1.0, beta=1.0):
    """Train the model for one epoch."""
    model.train()

    train_loss = 0
    bce_loss = 0
    kld_loss = 0

    device = next(model.parameters()).device

    for batch_idx, batch in enumerate(train_loader):
        data, labels = unpack_batch(batch)
        data = data.to(device)
        if labels is not None:
            labels = labels.to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = run_model(model, data, labels)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, alpha, beta)

        train_loss += loss.item()
        bce_loss += bce.item()
        kld_loss += kld.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item() / len(data):.6f}"
            )

    n = len(train_loader.dataset)
    return {
        "total": train_loss / n,
        "bce": (alpha * bce_loss) / n,
        "kld": (beta * kld_loss) / n
    }


def test_model(model, test_loader, alpha=1.0, beta=1.0):
    """Evaluate the model on the test set."""
    model.eval()
    device = next(model.parameters()).device

    test_loss = 0
    bce_loss = 0
    kld_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            data, labels = unpack_batch(batch)
            data = data.to(device)
            if labels is not None:
                labels = labels.to(device)

            recon_batch, mu, logvar = run_model(model, data, labels)
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


def build_model_config(dataset_name: str, input_dim: int, hidden_dim: int, latent_dim: int, learning_rate: float) -> dict[str, str | int | float]:
    """Build the config dictionary used to detect model-setting changes."""
    return {
        "dataset_name": dataset_name,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "learning_rate": learning_rate,
    }


def initialize_models(
    *,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    learning_rate: float,
    dataset_info: dict,
) -> None:
    """Create fresh VAE/CVAE models and optimizers for the current config."""
    vae_model = VAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(DEVICE)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

    cvae_model = None
    cvae_optimizer = None
    if dataset_info["has_labels"]:
        cvae_model = CVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_classes=dataset_info["num_classes"],
        ).to(DEVICE)
        cvae_optimizer = optim.Adam(cvae_model.parameters(), lr=learning_rate)

    st.session_state.vae_model = vae_model
    st.session_state.vae_optimizer = vae_optimizer
    st.session_state.cvae_model = cvae_model
    st.session_state.cvae_optimizer = cvae_optimizer
    st.session_state.history = empty_loss_history()
    st.session_state.test_history = empty_loss_history()
    st.session_state.cvae_history = empty_loss_history()
    st.session_state.cvae_test_history = empty_loss_history()
    st.session_state.test_metrics = None
    st.session_state.cvae_test_metrics = None
    st.session_state.trained = False
    st.session_state.random_samples = None
    st.session_state.conditional_samples = None
    st.session_state.selected_label_name = None


def sync_models_with_config(model_config: dict, dataset_info: dict) -> None:
    """Reinitialize models when the selected configuration changes."""
    if st.session_state.model_config == model_config:
        return

    initialize_models(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        learning_rate=model_config["learning_rate"],
        dataset_info=dataset_info,
    )
    st.session_state.model_config = model_config


def evaluate_if_available(model, test_loader, alpha: float, beta: float):
    """Return test metrics when a test loader exists, otherwise None."""
    if test_loader is None:
        return None
    return test_model(model, test_loader, alpha, beta)


def train_and_store_models(
    vae_model,
    vae_optimizer,
    cvae_model,
    cvae_optimizer,
    train_loader,
    test_loader,
    epochs: int,
    alpha: float,
    beta: float,
    image_shape,
    input_dim,
    preview_fn=None,
):
    """Train active models, show progress, and persist results in session state."""
    history = empty_loss_history()
    test_history = empty_loss_history()
    cvae_history = empty_loss_history()
    cvae_test_history = empty_loss_history()

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_container = st.container()

    for epoch in range(1, epochs + 1):
        vae_losses = train_model(vae_model, train_loader, vae_optimizer, epoch, alpha, beta)
        update_history(history, vae_losses)

        vae_test_losses = evaluate_if_available(vae_model, test_loader, alpha, beta)
        if vae_test_losses is not None:
            update_history(test_history, vae_test_losses)

        if cvae_model is not None and cvae_optimizer is not None:
            cvae_losses = train_model(
                cvae_model,
                train_loader,
                cvae_optimizer,
                epoch,
                alpha,
                beta,
            )
            update_history(cvae_history, cvae_losses)

            cvae_test_losses = evaluate_if_available(cvae_model, test_loader, alpha, beta)
            if cvae_test_losses is not None:
                update_history(cvae_test_history, cvae_test_losses)

        progress_bar.progress(epoch / epochs)
        status_text.info(
            f"Epoch {epoch}/{epochs} — Total loss: {vae_losses['total']:.4f}"
        )
        if preview_fn is not None:
            with preview_container:
                preview_fn(
                    epoch=epoch,
                    losses=vae_losses,
                    model=vae_model,
                    train_loader=train_loader,
                    image_shape=image_shape,
                    input_dim=input_dim,
                )

    st.session_state.history = history
    st.session_state.test_history = test_history
    st.session_state.cvae_history = cvae_history
    st.session_state.cvae_test_history = cvae_test_history
    st.session_state.test_metrics = evaluate_if_available(vae_model, test_loader, alpha, beta)
    st.session_state.cvae_test_metrics = evaluate_if_available(
        cvae_model,
        test_loader,
        alpha,
        beta,
    ) if cvae_model is not None else None
    st.session_state.trained = True
    st.session_state.random_samples = None
    st.session_state.conditional_samples = None


def generate_random_samples(vae_model, latent_dim: int):
    """Generate random decoded samples from the plain VAE latent space."""
    with torch.no_grad():
        latent_vectors = torch.randn(8, latent_dim).to(DEVICE)
        return vae_model.decode(latent_vectors).cpu()


def generate_conditional_samples(cvae_model, latent_dim: int, class_idx: int):
    """Generate decoded samples from the CVAE for a specific class."""
    with torch.no_grad():
        latent_vectors = torch.randn(8, latent_dim).to(DEVICE)
        class_labels = torch.full((8,), class_idx, dtype=torch.long, device=DEVICE)
        return cvae_model.decode(latent_vectors, class_labels).cpu()
