import streamlit as st


def empty_loss_history() -> dict[str, list[float]]:
    return {"Total": [], "Reconstruction": [], "Regularization (KL)": []}

def reset_model_state() -> None:
    """Reset all model-related values stored in Streamlit session state."""
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
    """Ensure all required session-state keys exist."""
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
    """Append one epoch of model losses to a history dictionary."""
    history["Total"].append(losses["total"])
    history["Reconstruction"].append(losses["bce"])
    history["Regularization (KL)"].append(losses["kld"])


def latest_metrics_from_history(history: dict[str, list[float]]) -> dict[str, float]:
    """Extract the latest values from a tracked loss history."""
    return {
        "total": history["Total"][-1],
        "bce": history["Reconstruction"][-1],
        "kld": history["Regularization (KL)"][-1],
    }
