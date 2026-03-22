import streamlit as st

from config import DEFAULT_DATASET
from state import initialize_session_state
from utils import load_dataset
from dl import train_and_store_models, sync_models_with_config, build_model_config
from ui import (
    render_header, 
    render_sidebar, 
    render_evaluation_section, 
    render_latent_space_section, 
    render_generation_section, 
    render_loss_analysis_section,
    render_epoch_preview  
)

st.set_page_config(page_title="VAE Project - Dashboard", layout="wide")

def main() -> None:
    """
    Orchestrate and run the Streamlit dashboard application.

    This function initializes the session state, renders the sidebar for 
    parameter selection, manages the dataset loading, synchronizes the 
    VAE/CVAE models with the selected configuration, and handles the 
    navigation between training and generation tabs.

    Returns
    -------
    None
        This function does not return any value; it side-effects by 
        rendering the Streamlit UI.
    """
    initialize_session_state()
    params = render_sidebar()

    # Chargement des données
    dataset_info = load_dataset(params["dataset_name"], params["batch_size"])
    train_loader = dataset_info["train_loader"]
    test_loader = dataset_info["test_loader"]
    input_dim = dataset_info["input_dim"]
    image_shape = dataset_info["image_shape"]

    # Config et synchro du modèle
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

    # Cette fonction crée les onglets Théorie, Entraînement et Génération
    render_header(
        dataset_info,
        params["alpha"],
        params["beta"],
        params["epochs"],
        params["hidden_dim"],
        params["latent_dim"],
    )

    # On accède à l'onglet via le session_state initialisé dans ui.py
    with st.session_state.tab_train:
        train_clicked = st.button("Launch training", use_container_width=True)
        
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
            st.success("Training finished.")

        if st.session_state.trained:
            render_evaluation_section(dataset_info, cvae_model)
            render_loss_analysis_section(dataset_info, cvae_model)
        else:
            st.info("Veuillez lancer l'entraînement pour voir l'analyse des pertes et les performances.")

    # LOGIQUE DE L'ONGLET GÉNÉRATION & LATENT
    if st.session_state.trained:
        # Ces fonctions utilisent 'with st.session_state.tab_gen' en interne
        render_generation_section(
            dataset_info,
            cvae_model,
            vae_model,
            image_shape,
            params["latent_dim"],
        )
        render_latent_space_section(vae_model, image_shape, params["latent_dim"])
    else:
        with st.session_state.tab_gen:
            st.warning("⚠️ L'exploration de l'espace latent et la génération seront disponibles après l'entraînement.")

if __name__ == "__main__":
    main()