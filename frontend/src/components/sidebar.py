"""Sidebar component for parameter configuration."""
from typing import Dict, Any, Optional
import streamlit as st


# Model hyperparameter configurations
MODEL_HYPERPARAMETERS = {
    "random_forest": {
        "n_estimators": {"min": 10, "max": 200, "default": 100, "step": 10},
        "max_depth": {"min": 1, "max": 50, "default": 10, "step": 1},
        "min_samples_split": {"min": 2, "max": 20, "default": 2, "step": 1},
    },
    "gradient_boosting": {
        "n_estimators": {"min": 10, "max": 200, "default": 100, "step": 10},
        "max_depth": {"min": 1, "max": 50, "default": 3, "step": 1},
        "min_samples_split": {"min": 2, "max": 20, "default": 2, "step": 1},
        "learning_rate": {"min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
    },
    "svm": {
        "C": {"min": 0.01, "max": 100.0, "default": 1.0, "step": 0.1},
    },
    "logistic_regression": {
        "C": {"min": 0.01, "max": 100.0, "default": 1.0, "step": 0.1},
    },
    "knn": {
        "k": {"min": 1, "max": 50, "default": 5, "step": 1},
    },
}


def render_sidebar(api_client) -> Dict[str, Any]:
    """Render sidebar with all parameter controls.
    
    Args:
        api_client: MLAPIClient instance for fetching datasets
        
    Returns:
        Dictionary containing all selected parameters
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset selection
    st.sidebar.subheader("Dataset")
    try:
        datasets = api_client.get_datasets()
        dataset_name = st.sidebar.selectbox(
            "Select Dataset",
            options=datasets,
            key="dataset_name"
        )
    except Exception as e:
        st.sidebar.error(f"Failed to load datasets: {str(e)}")
        dataset_name = None
    
    # Data split parameters
    st.sidebar.subheader("Data Split")
    test_size = st.sidebar.slider(
        "Test Split Ratio",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        key="test_size",
        help="Proportion of dataset to use for testing"
    )
    
    random_state = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
        key="random_state",
        help="Seed for reproducible data splitting"
    )
    
    # Model selection
    st.sidebar.subheader("Model")
    model_type = st.sidebar.selectbox(
        "Select Model",
        options=[
            "random_forest",
            "gradient_boosting",
            "svm",
            "logistic_regression",
            "knn"
        ],
        format_func=lambda x: x.replace("_", " ").title(),
        key="model_type"
    )
    
    # Hyperparameters (dynamic based on model)
    st.sidebar.subheader("Hyperparameters")
    hyperparameters = render_hyperparameters(model_type)
    
    # Action buttons
    st.sidebar.subheader("Actions")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        save_params = st.button(
            "💾 Save Params",
            key="save_params_btn",
            help="Save current hyperparameters to database"
        )
    
    with col2:
        clear_history = st.button(
            "🗑️ Clear History",
            key="clear_history_btn",
            help="Clear all experiment records"
        )
    
    return {
        "dataset_name": dataset_name,
        "test_size": test_size,
        "random_state": random_state,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "save_params": save_params,
        "clear_history": clear_history,
    }


def render_hyperparameters(model_type: str) -> Dict[str, Any]:
    """Render hyperparameter controls for selected model.
    
    Args:
        model_type: Type of model selected
        
    Returns:
        Dictionary of hyperparameter values
    """
    hyperparameters = {}
    
    if model_type not in MODEL_HYPERPARAMETERS:
        st.sidebar.warning(f"No hyperparameters defined for {model_type}")
        return hyperparameters
    
    params_config = MODEL_HYPERPARAMETERS[model_type]
    
    for param_name, config in params_config.items():
        # Format parameter name for display
        display_name = param_name.replace("_", " ").title()
        
        # Determine if parameter is float or int
        if isinstance(config["default"], float):
            value = st.sidebar.slider(
                display_name,
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=f"hyperparam_{param_name}",
                format="%.2f"
            )
        else:
            value = st.sidebar.slider(
                display_name,
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=f"hyperparam_{param_name}"
            )
        
        hyperparameters[param_name] = value
    
    return hyperparameters
