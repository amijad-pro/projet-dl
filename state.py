import streamlit as st


def empty_loss_history() -> dict[str, list[float]]:
    """
    Create a fresh dictionary structure to store training or testing losses.

    Returns
    -------
    dict[str, list[float]]
        A dictionary with keys "Total", "Reconstruction", and 
        "Regularization (KL)", each initialized with an empty list.
    """
    return {"Total": [], "Reconstruction": [], "Regularization (KL)": []}


def reset_model_state() -> None:
    """
    Reset all model-related values stored in the Streamlit session state.

    This function clears models, optimizers, histories, and generated 
    samples, effectively resetting the application to its initial state 
    before any training or configuration.

    Returns
    -------
    None
    """
    st.session_state.vae_model = None
    st.session_state.vae_optimizer = None
    st.session_state.cvae_model = None
    st.session_state.cvae_optimizer = None
    st.session_state.history = empty_loss_history()
    st.session_state.test_history = empty_loss_history()
    st.session_state.cvae_history = empty_loss_history()
    st.session_state.cvae_test_history = empty_loss_history()
    st.session_state.test_metrics = None
    st.session_state.cvae_test_metrics = None
    st.session_state.trained = False
    st.session_state.model_config = None
    st.session_state.random_samples = None
    st.session_state.conditional_samples = None
    st.session_state.selected_label_name = None


def initialize_session_state() -> None:
    """
    Ensure all required session-state keys exist upon application startup.

    If the 'vae_model' key is missing, it triggers a full reset of the 
    model state. It also sets default values for UI-related state keys 
    if they are not already present.

    Returns
    -------
    None
    """
    if "vae_model" not in st.session_state:
        reset_model_state()

    defaults = {
        "random_samples": None,
        "conditional_samples": None,
        "selected_label_name": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def update_history(history: dict[str, list[float]], losses: dict[str, float]) -> None:
    """
    Append one epoch of model losses to a history tracking dictionary.

    Parameters
    ----------
    history : dict[str, list[float]]
        The history dictionary to update (modified in-place).
    losses : dict[str, float]
        A dictionary containing the 'total', 'bce', and 'kld' values 
        for the current epoch.

    Returns
    -------
    None
    """
    history["Total"].append(losses["total"])
    history["Reconstruction"].append(losses["bce"])
    history["Regularization (KL)"].append(losses["kld"])


def latest_metrics_from_history(history: dict[str, list[float]]) -> dict[str, float]:
    """
    Extract the most recent loss values from a tracked history dictionary.

    Parameters
    ----------
    history : dict[str, list[float]]
        A history dictionary containing lists of previous losses.

    Returns
    -------
    dict[str, float]
        A dictionary containing the last 'total', 'bce', and 'kld' 
        values recorded.

    Raises
    ------
    IndexError
        If the history lists are empty.
    """
    return {
        "total": history["Total"][-1],
        "bce": history["Reconstruction"][-1],
        "kld": history["Regularization (KL)"][-1],
    }
