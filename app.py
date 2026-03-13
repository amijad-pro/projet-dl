
#streamlit run app.py

"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""
import streamlit as st  
import torch
import torch.optim as optim
from dl import VAE, CVAE, train_model, test_model
import matplotlib.pyplot as plt
import numpy as np
from functions import (
    load_dataset,
    plot_reconstructions,
    plot_losses,
    plot_latent_space,
    plot_sample_grid,
    render_model_explanations,
    render_latent_space_explanation,
)

# 1. CONFIGURATION INITIALE
st.set_page_config(page_title="VAE Project - Dashboard")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_model_state():
    st.session_state.vae_model = None
    st.session_state.vae_optimizer = None
    st.session_state.cvae_model = None
    st.session_state.cvae_optimizer = None
    st.session_state.history = None
    st.session_state.test_history = None
    st.session_state.test_metrics = None
    st.session_state.cvae_test_metrics = None
    st.session_state.trained = False
    st.session_state.model_config = None
    st.session_state.random_samples = None
    st.session_state.conditional_samples = None


if "vae_model" not in st.session_state:
    reset_model_state()

if "random_samples" not in st.session_state:
    st.session_state.random_samples = None

if "conditional_samples" not in st.session_state:
    st.session_state.conditional_samples = None

if "selected_label_name" not in st.session_state:
    st.session_state.selected_label_name = None


# 2. BARRE LATÉRALE (Paramètres)
st.sidebar.title("Dataset Navigation")

dataset_name = st.sidebar.radio(
    "Choose a dataset",
    ["MNIST", "FashionMNIST", "Frey Faces"],
)

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
    # Always create the plain VAE
    vae_model = VAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    vae_optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
    
    cvae_model = None
    cvae_optimizer = None
    if dataset_info["has_labels"]:
        cvae_model = CVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_classes=dataset_info["num_classes"],
        ).to(device)
        cvae_optimizer = optim.Adam(cvae_model.parameters(), lr=learning_rate)

    st.session_state.vae_model = vae_model
    st.session_state.vae_optimizer = vae_optimizer
    st.session_state.cvae_model = cvae_model
    st.session_state.cvae_optimizer = cvae_optimizer
    
    st.session_state.history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }
    st.session_state.test_history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }

    st.session_state.test_metrics = None
    st.session_state.cvae_test_metrics = None
    st.session_state.trained = False
    st.session_state.random_samples = None
    st.session_state.conditional_samples = None
    st.session_state.selected_label_name = None
    st.session_state.model_config = model_config

vae_model = st.session_state.vae_model
vae_optimizer = st.session_state.vae_optimizer
cvae_model = st.session_state.cvae_model
cvae_optimizer = st.session_state.cvae_optimizer


# 4. INTERFACE PRINCIPALE

st.title(f"VAE Dashboard — {dataset_info['title']}")
st.write(dataset_info["description"])
render_model_explanations(dataset_info)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Alpha", alpha)
col2.metric("Beta", beta)
col3.metric("Epochs", epochs)
col4.metric("Hidden dim", hidden_dim)
col5.metric("Latent dim", latent_dim)

st.divider()

train_clicked = st.button("Launch training", use_container_width=True)


# Reserve positions in the layout so the sections render in the order we want.
generation_section = st.container()
loss_analysis_section = st.container()
latent_space_section = st.container()

st.divider()

## Training 


if train_clicked:
    history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }

    test_history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }

    cvae_history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }

    cvae_test_history = {
        "Total": [],
        "Reconstruction": [],
        "Regularization (KL)": [],
    }

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_container = st.container()

    for epoch in range(1, epochs + 1):
        losses = train_model(vae_model, train_loader, vae_optimizer, epoch, alpha, beta)

        history["Total"].append(losses["total"])
        history["Reconstruction"].append(losses["bce"])
        history["Regularization (KL)"].append(losses["kld"])
        
        if test_loader is not None:
            test_losses = test_model(vae_model, test_loader, alpha, beta)
            test_history["Total"].append(test_losses["total"])
            test_history["Reconstruction"].append(test_losses["bce"])
            test_history["Regularization (KL)"].append(test_losses["kld"])

        if cvae_model is not None:
            cvae_losses = train_model(cvae_model, train_loader, cvae_optimizer, epoch, alpha, beta)

            cvae_history["Total"].append(cvae_losses["total"])
            cvae_history["Reconstruction"].append(cvae_losses["bce"])
            cvae_history["Regularization (KL)"].append(cvae_losses["kld"])

            if test_loader is not None:
                cvae_test_losses = test_model(cvae_model, test_loader, alpha, beta)
                cvae_test_history["Total"].append(cvae_test_losses["total"])
                cvae_test_history["Reconstruction"].append(cvae_test_losses["bce"])
                cvae_test_history["Regularization (KL)"].append(cvae_test_losses["kld"])

        progress_bar.progress(epoch / epochs)
        status_text.info(
            f"Epoch {epoch}/{epochs} — Total loss: {losses['total']:.4f}"
        )

        with preview_container:
            st.subheader(f"Reconstructions after epoch {epoch}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total", f"{losses['total']:.4f}")
            col2.metric("Reconstruction", f"{losses['bce']:.4f}")
            col3.metric("KL", f"{losses['kld']:.4f}")           
            
            plot_reconstructions(
                model=vae_model,
                dataset=train_loader.dataset,
                image_shape=image_shape,
                input_dim=input_dim,
                n=8,
            )

    st.session_state.history = history
    st.session_state.test_history = test_history
    st.session_state.cvae_history = cvae_history
    st.session_state.cvae_test_history = cvae_test_history
    st.session_state.trained = True
    st.session_state.random_samples = None
    st.session_state.conditional_samples = None


    if test_loader is not None:
        st.session_state.test_metrics = test_model(vae_model, test_loader, alpha, beta)

        if cvae_model is not None:
            st.session_state.cvae_test_metrics = test_model(cvae_model, test_loader, alpha, beta)
        else:
            st.session_state.cvae_test_metrics = None
    else:
        st.session_state.test_metrics = None
        st.session_state.cvae_test_metrics = None

## Display trained results 

if st.session_state.trained:
    st.success("Training finished.")

    with generation_section:
        st.divider()
        st.header("Generated Samples")

        generate_clicked = st.button("Generate new samples", use_container_width=True)

        st.markdown(
            "The plain **VAE** samples a random vector from latent space and decodes it into an image. "
            "The **CVAE** also conditions on a label, which lets you request a specific class."
        )

        if generate_clicked:
            with torch.no_grad():
                z = torch.randn(8, latent_dim).to(device)
                st.session_state.random_samples = vae_model.decode(z).cpu()

        if st.session_state.random_samples is not None:
            st.subheader("Random generation (plain VAE)")
            plot_sample_grid(
                st.session_state.random_samples,
                image_shape=image_shape,
            )

        if dataset_info["has_labels"] and cvae_model is not None:
            st.subheader("Conditional generation (CVAE)")
            st.write("Choose a label, then click the button to generate conditioned samples.")

            if (
                st.session_state.selected_label_name is None
                or st.session_state.selected_label_name not in dataset_info["class_names"]
            ):
                st.session_state.selected_label_name = dataset_info["class_names"][0]

            with st.form("conditional_generation_form"):
                selected_label_name = st.selectbox(
                    "Generate a sample of:",
                    options=dataset_info["class_names"],
                    index=dataset_info["class_names"].index(st.session_state.selected_label_name),
                    key="conditional_label_select",
                )

                conditional_clicked = st.form_submit_button("Generate conditioned samples")

            st.session_state.selected_label_name = selected_label_name
            class_idx = dataset_info["class_names"].index(selected_label_name)

            if conditional_clicked:
                with torch.no_grad():
                    z = torch.randn(8, latent_dim).to(device)
                    y = torch.full((8,), class_idx, dtype=torch.long, device=device)
                    st.session_state.conditional_samples = cvae_model.decode(z, y).cpu()

            if st.session_state.conditional_samples is not None:
                plot_sample_grid(
                    st.session_state.conditional_samples,
                    image_shape=image_shape,
                )

    st.divider()
    st.header("Test Evaluation")

    final_train_total = st.session_state.history["Total"][-1]
    final_train_recon = st.session_state.history["Reconstruction"][-1]
    final_train_kl = st.session_state.history["Regularization (KL)"][-1]

    st.subheader("Final Training Losses")
    train_col1, train_col2, train_col3 = st.columns(3)
    train_col1.metric("Train total loss", f"{final_train_total:.4f}")
    train_col2.metric("Train reconstruction loss", f"{final_train_recon:.4f}")
    train_col3.metric("Train KL loss", f"{final_train_kl:.4f}")

    st.subheader("Test Losses")
    if st.session_state.test_metrics is None:
        st.info("No test loader available for this dataset.")
    else:
        test_col1, test_col2, test_col3 = st.columns(3)
        test_col1.metric("Test total loss", f"{st.session_state.test_metrics['total']:.4f}")
        test_col2.metric("Test reconstruction loss", f"{st.session_state.test_metrics['bce']:.4f}")
        test_col3.metric("Test KL loss", f"{st.session_state.test_metrics['kld']:.4f}")
    
    if dataset_info["has_labels"] and cvae_model is not None and st.session_state.cvae_history["Total"]:
        final_cvae_total = st.session_state.cvae_history["Total"][-1]
        final_cvae_recon = st.session_state.cvae_history["Reconstruction"][-1]
        final_cvae_kl = st.session_state.cvae_history["Regularization (KL)"][-1]

        st.subheader("Final Training Losses — CVAE")
        ctrain_col1, ctrain_col2, ctrain_col3 = st.columns(3)
        ctrain_col1.metric("Train total loss", f"{final_cvae_total:.4f}")
        ctrain_col2.metric("Train reconstruction loss", f"{final_cvae_recon:.4f}")
        ctrain_col3.metric("Train KL loss", f"{final_cvae_kl:.4f}")

        st.subheader("Test Losses — CVAE")
        if st.session_state.cvae_test_metrics is not None:
            ctest_col1, ctest_col2, ctest_col3 = st.columns(3)
            ctest_col1.metric("Test total loss", f"{st.session_state.cvae_test_metrics['total']:.4f}")
            ctest_col2.metric("Test reconstruction loss", f"{st.session_state.cvae_test_metrics['bce']:.4f}")
            ctest_col3.metric("Test KL loss", f"{st.session_state.cvae_test_metrics['kld']:.4f}")

    with latent_space_section:
        st.divider()
        st.header("Latent Space Exploration")
        render_latent_space_explanation()
        plot_latent_space(
            model=vae_model,
            image_shape=image_shape,
            latent_dim=latent_dim,
            n=12,
        )

    with loss_analysis_section:
        st.divider()
        st.header("Loss Analysis")
        plot_losses(st.session_state.history, st.session_state.test_history)

        if dataset_info["has_labels"] and cvae_model is not None and st.session_state.cvae_history["Total"]:
            st.subheader("CVAE Loss Analysis")
            plot_losses(st.session_state.cvae_history, st.session_state.cvae_test_history)

else:
    st.info("Train a model first to view evaluation, latent space, and generation.")
