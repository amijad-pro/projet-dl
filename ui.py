import streamlit as st
from plotting import plot_reconstructions, plot_sample_grid, plot_latent_space, plot_losses
from dl import generate_random_samples, generate_conditional_samples
from state import latest_metrics_from_history
from config import DATASET_OPTIONS


def render_sidebar() -> dict[str, float | int | str]:
    """Render sidebar controls and return the selected training parameters."""
    st.sidebar.title("Dataset Navigation")

    dataset_name = st.sidebar.radio("Choose a dataset", DATASET_OPTIONS)

    st.sidebar.divider()
    st.sidebar.subheader("Training Parameters")

    return {
        "dataset_name": dataset_name,
        "alpha": st.sidebar.slider("Alpha", 0.0, 5.0, 1.0, 0.1),
        "beta": st.sidebar.slider("Beta", 0.0, 10.0, 1.0, 0.1),
        "epochs": st.sidebar.slider("Number of epochs", 1, 50, 5),
        "hidden_dim": st.sidebar.slider(
            "Hidden dimension",
            min_value=32,
            max_value=1024,
            value=400,
            step=32,
        ),
        "latent_dim": st.sidebar.slider(
            "Latent dimension",
            min_value=2,
            max_value=128,
            value=2,
            step=2,
        ),
        "batch_size": st.sidebar.select_slider(
            "Batch size",
            options=[32, 64, 128, 256],
            value=128,
        ),
        "learning_rate": st.sidebar.slider(
            "Learning rate",
            min_value=1e-5,
            max_value=1e-2,
            value=1e-3,
            step=1e-5,
            format="%.5f",
        ),
    }


def render_header(dataset_info: dict, alpha: float, beta: float, epochs: int, hidden_dim: int, latent_dim: int) -> None:
    """Render the main page title and initialize the 3-tab segmentation."""
    st.title(f"VAE Dashboard — {dataset_info['title']}")
    
    # Création des onglets pour la segmentation en 3 parties
    tab_theory, tab_train, tab_gen = st.tabs(["📚 Théorie", "📊 Entraînement", "🎨 Génération & Espace Latent"]) 
    
    st.session_state.tab_theory = tab_theory
    st.session_state.tab_train = tab_train
    st.session_state.tab_gen = tab_gen

    with tab_theory:
        st.write(dataset_info["description"])
        render_model_explanations(dataset_info)

    with tab_train:
        st.subheader("Active Configuration")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Alpha", alpha)
        metric_cols[1].metric("Beta", beta)
        metric_cols[2].metric("Epochs", epochs)
        metric_cols[3].metric("Hidden dim", hidden_dim)
        metric_cols[4].metric("Latent dim", latent_dim)
        st.divider()


def render_epoch_preview(*, epoch, losses, model, train_loader, image_shape, input_dim) -> None:
    """Render metrics and reconstructions in the Training tab."""
    with st.session_state.tab_train:
        st.subheader(f"Reconstructions after epoch {epoch}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Total", f"{losses['total']:.4f}")
        metric_cols[1].metric("Reconstruction", f"{losses['bce']:.4f}")
        metric_cols[2].metric("KL", f"{losses['kld']:.4f}")

        plot_reconstructions(
            model=model,
            dataset=train_loader.dataset,
            image_shape=image_shape,
            input_dim=input_dim,
            n=8,
        )


def render_generation_section(dataset_info, cvae_model, vae_model, image_shape, latent_dim) -> None:
    """Render sample generation in the Generation tab."""
    with st.session_state.tab_gen:
        st.header("Generated Samples")
        generate_clicked = st.button("Generate new samples", use_container_width=True)
        st.markdown(
            "The plain **VAE** samples a random vector from latent space and decodes it into an image. "
            "The **CVAE** also conditions on a label, which lets you request a specific class."
        )

        if generate_clicked:
            st.session_state.random_samples = generate_random_samples(vae_model, latent_dim)

        if st.session_state.random_samples is not None:
            st.subheader("Random generation (plain VAE)")
            plot_sample_grid(st.session_state.random_samples, image_shape=image_shape)

        if not dataset_info["has_labels"] or cvae_model is None:
            return

        st.subheader("Conditional generation (CVAE)")
        st.write("Choose a label, then click the button to generate conditioned samples.")

        class_names = dataset_info["class_names"]
        if st.session_state.selected_label_name not in class_names:
            st.session_state.selected_label_name = class_names[0]

        with st.form("conditional_generation_form"):
            selected_label_name = st.selectbox(
                "Generate a sample of:",
                options=class_names,
                index=class_names.index(st.session_state.selected_label_name),
                key="conditional_label_select",
            )
            conditional_clicked = st.form_submit_button("Generate conditioned samples")

        st.session_state.selected_label_name = selected_label_name
        selected_class_idx = class_names.index(selected_label_name)

        if conditional_clicked:
            st.session_state.conditional_samples = generate_conditional_samples(
                cvae_model, latent_dim, selected_class_idx,
            )

        if st.session_state.conditional_samples is not None:
            plot_sample_grid(
                st.session_state.conditional_samples,
                image_shape=image_shape,
            )


def render_evaluation_section(dataset_info, cvae_model) -> None:
    """Render final evaluation in the Training tab."""
    with st.session_state.tab_train:
        st.divider()
        st.header("VAE Test Evaluation")
        render_final_loss_metrics("VAE Final Training Losses", latest_metrics_from_history(st.session_state.history))

        if st.session_state.test_metrics is None:
            st.info("No test loader available for this dataset.")
        else:
            render_final_loss_metrics("Test Losses", st.session_state.test_metrics)

        # Affichage des métriques CVAE si disponibles
        has_cvae_results = (
            dataset_info["has_labels"]
            and cvae_model is not None
            and bool(st.session_state.cvae_history["Total"])
        )
        if has_cvae_results:
            render_final_loss_metrics(
                "Final Training Losses - CVAE",
                latest_metrics_from_history(st.session_state.cvae_history),
            )
            if st.session_state.cvae_test_metrics is not None:
                render_final_loss_metrics("Test Losses - CVAE", st.session_state.cvae_test_metrics)


def render_latent_space_section(vae_model, image_shape, latent_dim) -> None:
    """Render latent space in the Generation tab."""
    with st.session_state.tab_gen:
        st.divider()
        st.header("Latent Space Exploration")
        render_latent_space_explanation()
        plot_latent_space(model=vae_model, image_shape=image_shape, latent_dim=latent_dim, n=12)


def render_loss_analysis_section(dataset_info: dict, cvae_model) -> None:
    """Render training and test loss charts for available models in the Training tab."""
    with st.session_state.tab_train:
        st.divider()
        st.header("VAE Loss Analysis")
        # Plot VAE classique
        plot_losses(st.session_state.history, st.session_state.test_history)

        # Plot CVAE si disponible
        has_cvae_results = (
            dataset_info["has_labels"]
            and cvae_model is not None
            and bool(st.session_state.cvae_history["Total"])
        )
        if has_cvae_results:
            st.subheader("CVAE Loss Analysis")
            plot_losses(st.session_state.cvae_history, st.session_state.cvae_test_history)


# --- FONCTIONS DE CONTENU ---

def render_model_explanations(dataset_info):
    st.markdown(r"""
    ### What this model is doing
    A **Variational Autoencoder (VAE)** learns how to compress an image into a
    small latent representation, then reconstruct it. The **encoder** maps an
    input image $x$ to the parameters of a latent Gaussian distribution, and the
    **decoder** reconstructs an image from a sampled latent vector $z$.

    The encoder predicts:
    - $\mu$: the mean of the latent distribution
    - $\log(\sigma^2)$: the log-variance of the latent distribution

    A latent vector is then sampled and passed to the decoder. This is what lets
    the model both reconstruct inputs and generate brand-new samples.
    """)
    _render_parameter_explanation()
    _render_dataset_specific_info(dataset_info)

def render_final_loss_metrics(title, metrics):
    st.subheader(title)
    cols = st.columns(3)
    cols[0].metric("Total loss", f"{metrics['total']:.4f}")
    cols[1].metric("Reconstruction loss", f"{metrics['bce']:.4f}")
    cols[2].metric("KL loss", f"{metrics['kld']:.4f}")

def render_latent_space_explanation():
    st.markdown("""
    The **latent space** is the compressed bottleneck learned by the encoder.
    Each point in this space corresponds to a decoded image.

    **What to look for:**
    - nearby points should generate similar images
    - moving smoothly across the grid should gradually change the output
    - well-trained models often show structured transitions between styles, shapes, or classes
    """)

def _render_parameter_explanation():
    with st.expander("How the training parameters affect the model", expanded=True):
        st.markdown(r"""
    The VAE is trained with a weighted version of the usual ELBO objective:

    $$ \mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{recon}} + \beta D_{KL}\big(q_\phi(z \mid x) \, \| \, p(z)\big) $$

    **Parameters details:**
    - **Alpha ($\alpha$)** controls the weight of the reconstruction term.
    - **Beta ($\beta$)** controls the weight of the KL term.
    - **Epochs** = number of full passes through the training set.
    - **Hidden dimension** = width of internal layers.
    - **Latent dimension** = size of the bottleneck $z$.
    """)

def _render_dataset_specific_info(dataset_info):
    if dataset_info["has_labels"]:
        st.info("On labeled datasets such as MNIST and FashionMNIST, the app trains both a standard VAE and a CVAE.")
    else:
        st.info("For unlabeled datasets, the app trains a standard VAE.")