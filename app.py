
#streamlit run app.py

"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""
import streamlit as st  
import torch
import torch.optim as optim
from dl import VAE, train_model, test_model
from utils import get_mnist_loaders, get_frey_loader, get_fashion_mnist_loaders
import matplotlib.pyplot as plt
import os
import numpy as np
from functions import load_dataset, get_image_from_dataset, plot_reconstructions, plot_losses, plot_latent_space, plot_generated_samples

# 1. CONFIGURATION INITIALE
st.set_page_config(page_title="VAE Project - Dashboard")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_model_state():
    st.session_state.model = None
    st.session_state.optimizer = None
    st.session_state.history = None
    st.session_state.test_loss = None
    st.session_state.trained = False
    st.session_state.model_config = None

if "model" not in st.session_state:
    reset_model_state()

# 2. BARRE LATÉRALE (Paramètres)
st.sidebar.title("Dataset Navigation")

dataset_name = st.sidebar.radio(
    "Choose a dataset",
    ["MNIST", "FashionMNIST", "Frey Faces"],
)

if dataset_name in ["MNIST", "FashionMNIST"]:
    input_dim = 28 * 28
elif dataset_name == "FreyFaces":
    input_dim = 20 * 28

st.sidebar.divider()
st.sidebar.subheader("Training Parameters")

alpha = st.sidebar.slider("Alpha", 0.0, 5.0, 1.0, 0.1)
beta = st.sidebar.slider("Beta", 0.0, 10.0, 1.0, 0.1)
epochs = st.sidebar.slider("Number of epochs", 1, 50, 5)
hidden_dim = st.sidebar.slider("Hidden dimension", 32, 1024, 400, step=32)
latent_dim = st.sidebar.slider("Latent dimension", 2, 128, 2, step=2)
batch_size = st.sidebar.select_slider( "Batch size", options=[32, 64, 128, 256], value=128,)
learning_rate = st.sidebar.slider("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, step=1e-5, format="%.5f")


# 3. CHARGEMENT DES DONNÉES ET DU MODÈLE
dataset_info = load_dataset(dataset_name, batch_size)
train_loader = dataset_info["train_loader"]
test_loader = dataset_info["test_loader"]
input_dim = dataset_info["input_dim"]
image_shape = dataset_info["image_shape"]

model_config = {
    "dataset_name": dataset_name,
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "latent_dim": latent_dim,
    "learning_rate": learning_rate,
}

if st.session_state.model_config != model_config:
    model = VAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    st.session_state.model = model
    st.session_state.optimizer = optimizer
    st.session_state.history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }
    st.session_state.test_loss = None
    st.session_state.trained = False
    st.session_state.model_config = model_config


model = st.session_state.model
optimizer = st.session_state.optimizer


# 4. INTERFACE PRINCIPALE

st.title(f"VAE Dashboard — {dataset_info['title']}")
st.write(dataset_info["description"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Alpha", alpha)
col2.metric("Beta", beta)
col3.metric("Epochs", epochs)
col4.metric("Hidden dim", hidden_dim)
col5.metric("Latent dim", latent_dim)

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    train_clicked = st.button("Launch training", use_container_width=True)
with col_b:
    generate_clicked = st.button("Generate new samples", use_container_width=True)

st.divider()

## Training 


if train_clicked:
    history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_container = st.container()

    for epoch in range(1, epochs + 1):
        losses = train_model(model, train_loader, optimizer, epoch, alpha, beta)

        history["Total"].append(losses["total"])
        history["Reconstruction"].append(losses["bce"])
        history["Regularization (KL)"].append(losses["kld"])

        progress_bar.progress(epoch / epochs)
        status_text.info(
            f"Epoch {epoch}/{epochs} — Total loss: {losses['total']:.4f}"
        )

        with preview_container:
            st.subheader(f"Reconstructions after epoch {epoch}")
            plot_reconstructions(
                model=model,
                dataset=train_loader.dataset,
                image_shape=image_shape,
                input_dim=input_dim,
                n=8,
            )

    st.session_state.history = history
    st.session_state.trained = True

    if test_loader is not None:
        st.session_state.test_loss = test_model(model, test_loader, alpha, beta)
    else:
        st.session_state.test_loss = None

## Display trained results 

if st.session_state.trained:
    st.success("Training finished.")

    st.divider()
    st.header("Test Evaluation")

    if st.session_state.test_loss is None:
        st.info("No test loader available for this dataset.")
    else:
        st.metric("Test loss", f"{st.session_state.test_loss:.4f}")

    st.divider()
    st.header("Loss Analysis")
    plot_losses(st.session_state.history)

    st.divider()
    st.header("Latent Space Exploration")
    plot_latent_space(
        model=model,
        image_shape=image_shape,
        latent_dim=latent_dim,
        n=12,
    )
else:
    st.info("Train a model first to view evaluation, latent space, and generation.")


if generate_clicked:
    if not st.session_state.trained:
        st.warning("Please train the model first.")
    else:
        st.header("Generated Samples")
        plot_generated_samples(
            model=model,
            image_shape=image_shape,
            latent_dim=latent_dim,
            n=8,
        )