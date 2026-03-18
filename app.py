#streamlit run app.py

"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""
import streamlit as st

from config import DEFAULT_DATASET
from state import initialize_session_state
from utils import load_dataset
from dl import train_and_store_models, sync_models_with_config, build_model_config
from ui import render_header, render_sidebar, render_evaluation_section, render_latent_space_section, render_generation_section, render_loss_analysis_section


st.set_page_config(page_title="VAE Project - Dashboard")


def main() -> None:
    """Run the Streamlit application."""
    initialize_session_state()
    params = render_sidebar()

    dataset_info = load_dataset(params["dataset_name"], params["batch_size"])
    train_loader = dataset_info["train_loader"]
    test_loader = dataset_info["test_loader"]
    input_dim = dataset_info["input_dim"]
    image_shape = dataset_info["image_shape"]

    model_config = build_model_config(
        dataset_name=params["dataset_name"],
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
        learning_rate=params["learning_rate"],
    )
    sync_models_with_config(model_config, dataset_info)

    vae_model = st.session_state.vae_model
    vae_optimizer = st.session_state.vae_optimizer
    cvae_model = st.session_state.cvae_model
    cvae_optimizer = st.session_state.cvae_optimizer

    render_header(
        dataset_info,
        params["alpha"],
        params["beta"],
        params["epochs"],
        params["hidden_dim"],
        params["latent_dim"],
    )

    train_clicked = st.button("Launch training", use_container_width=True)

    generation_section = st.container()
    loss_analysis_section = st.container()
    latent_space_section = st.container()

    st.divider()

    if train_clicked:
        train_and_store_models(
            vae_model=vae_model,
            vae_optimizer=vae_optimizer,
            cvae_model=cvae_model,
            cvae_optimizer=cvae_optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=params["epochs"],
            alpha=params["alpha"],
            beta=params["beta"],
            image_shape=image_shape,
            input_dim=input_dim,
        )

    if not st.session_state.trained:
        st.info("Train a model first to view evaluation, latent space, and generation.")
        return

    st.success("Training finished.")

    with generation_section:
        render_generation_section(
            dataset_info,
            cvae_model,
            vae_model,
            image_shape,
            params["latent_dim"],
        )

    render_evaluation_section(dataset_info, cvae_model)

    with latent_space_section:
        render_latent_space_section(vae_model, image_shape, params["latent_dim"])

    with loss_analysis_section:
        render_loss_analysis_section(dataset_info, cvae_model)


if __name__ == "__main__":
    main()
