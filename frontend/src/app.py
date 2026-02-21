"""Main Streamlit application for ML Dashboard."""
import streamlit as st
from typing import Optional, Dict, Any
import time

from src.config import settings
from src.api_client import MLAPIClient
from src.components.sidebar import render_sidebar
from src.components.metrics import render_metrics_cards
from src.components.visualizations import (
    render_data_preview,
    render_confusion_matrix,
    render_feature_importance,
    render_experiment_history,
    render_accuracy_trend,
    render_classification_report
)


# Page configuration
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = MLAPIClient()
    
    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = None
    
    if "last_training_result" not in st.session_state:
        st.session_state.last_training_result = None
    
    if "experiment_history" not in st.session_state:
        st.session_state.experiment_history = []
    
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = None
    
    if "preview_data" not in st.session_state:
        st.session_state.preview_data = None


def load_dataset_info(api_client: MLAPIClient, dataset_name: str):
    """Load dataset information and preview.
    
    Args:
        api_client: API client instance
        dataset_name: Name of dataset to load
    """
    try:
        # Get dataset info
        dataset_info = api_client.get_dataset(dataset_name)
        st.session_state.dataset_info = dataset_info
        
        # Get preview
        preview_data = api_client.get_dataset_preview(dataset_name, n_rows=10)
        st.session_state.preview_data = preview_data
        
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        st.session_state.dataset_info = None
        st.session_state.preview_data = None


def train_model(
    api_client: MLAPIClient,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Train model with given configuration.
    
    Args:
        api_client: API client instance
        config: Training configuration
        
    Returns:
        Training result or None on failure
    """
    try:
        result = api_client.train_model(config)
        return result
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None


def save_experiment(
    api_client: MLAPIClient,
    config: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None
):
    """Save experiment to database.
    
    Args:
        api_client: API client instance
        config: Training configuration
        result: Training result (optional)
    """
    try:
        experiment_data = {
            "dataset_name": config["dataset_name"],
            "model_type": config["model_type"],
            "hyperparameters": config["hyperparameters"],
            "accuracy": result["accuracy"] if result else 0.0,
            "f1_score": result["f1_score"] if result else 0.0,
            "training_time": result.get("training_time", 0.0) if result else 0.0
        }
        
        api_client.save_experiment(experiment_data)
        st.success("✅ Experiment saved successfully!")
        
        # Refresh experiment history
        load_experiment_history(api_client)
        
    except Exception as e:
        st.error(f"Failed to save experiment: {str(e)}")


def load_experiment_history(api_client: MLAPIClient):
    """Load experiment history from database.
    
    Args:
        api_client: API client instance
    """
    try:
        experiments = api_client.get_experiments()
        st.session_state.experiment_history = experiments
    except Exception as e:
        st.error(f"Failed to load experiment history: {str(e)}")
        st.session_state.experiment_history = []


def clear_experiment_history(api_client: MLAPIClient):
    """Clear all experiment history.
    
    Args:
        api_client: API client instance
    """
    try:
        api_client.clear_experiments()
        st.session_state.experiment_history = []
        st.success("✅ Experiment history cleared!")
    except Exception as e:
        st.error(f"Failed to clear history: {str(e)}")


def export_model(api_client: MLAPIClient, model_id: str):
    """Export trained model.
    
    Args:
        api_client: API client instance
        model_id: Model identifier
    """
    try:
        model_bytes = api_client.export_model(model_id)
        
        st.download_button(
            label="💾 Download Model",
            data=model_bytes,
            file_name=f"model_{model_id}.pkl",
            mime="application/octet-stream",
            help="Download trained model as pickle file"
        )
        
    except Exception as e:
        st.error(f"Failed to export model: {str(e)}")


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    api_client = st.session_state.api_client
    
    # Title
    st.title(f"{settings.app_icon} {settings.app_title}")
    st.markdown("Train and evaluate machine learning models on scikit-learn datasets")
    
    # Render sidebar and get configuration
    sidebar_config = render_sidebar(api_client)
    
    # Handle sidebar actions
    if sidebar_config["save_params"]:
        # Save current parameters
        save_experiment(
            api_client,
            {
                "dataset_name": sidebar_config["dataset_name"],
                "model_type": sidebar_config["model_type"],
                "hyperparameters": sidebar_config["hyperparameters"],
                "test_size": sidebar_config["test_size"],
                "random_state": sidebar_config["random_state"]
            }
        )
    
    if sidebar_config["clear_history"]:
        clear_experiment_history(api_client)
    
    # Load dataset info if dataset changed
    if sidebar_config["dataset_name"]:
        if (st.session_state.dataset_info is None or 
            st.session_state.dataset_info.get("dataset_name") != sidebar_config["dataset_name"]):
            load_dataset_info(api_client, sidebar_config["dataset_name"])
    
    # Load experiment history on first run
    if not st.session_state.experiment_history:
        load_experiment_history(api_client)
    
    # Main content area
    st.divider()
    
    # Training controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        train_button = st.button(
            "🚀 Start Training",
            type="primary",
            disabled=sidebar_config["dataset_name"] is None,
            help="Train model with current configuration"
        )
    
    with col2:
        export_button = st.button(
            "📦 Export Model",
            disabled=st.session_state.current_model_id is None,
            help="Export trained model as pickle file"
        )
    
    # Handle training
    if train_button:
        if sidebar_config["dataset_name"] is None:
            st.error("Please select a dataset first")
        else:
            # Show progress
            with st.spinner("Training model..."):
                progress_bar = st.progress(0)
                
                # Prepare training config
                training_config = {
                    "dataset_name": sidebar_config["dataset_name"],
                    "test_size": sidebar_config["test_size"],
                    "random_state": sidebar_config["random_state"],
                    "model_type": sidebar_config["model_type"],
                    "hyperparameters": sidebar_config["hyperparameters"]
                }
                
                # Simulate progress
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Train model
                result = train_model(api_client, training_config)
                
                if result:
                    st.session_state.last_training_result = result
                    st.session_state.current_model_id = result.get("model_id")
                    
                    st.success("✅ Training completed successfully!")
                    
                    # Auto-save experiment
                    save_experiment(api_client, training_config, result)
                    
                    # Rerun to update UI
                    st.rerun()
    
    # Handle export
    if export_button and st.session_state.current_model_id:
        export_model(api_client, st.session_state.current_model_id)
    
    st.divider()
    
    # Metrics cards
    result = st.session_state.last_training_result
    render_metrics_cards(
        accuracy=result.get("accuracy") if result else None,
        f1_score=result.get("f1_score") if result else None,
        model_name=sidebar_config["model_type"] if result else None,
        dataset_info=st.session_state.dataset_info
    )
    
    st.divider()
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Preview",
        "📊 Model Performance",
        "📈 Experiment History",
        "📄 Details"
    ])
    
    with tab1:
        render_data_preview(st.session_state.preview_data)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            render_confusion_matrix(
                confusion_matrix=result.get("confusion_matrix") if result else None,
                target_names=st.session_state.dataset_info.get("target_names") if st.session_state.dataset_info else None
            )
        
        with col2:
            render_feature_importance(
                feature_importances=result.get("feature_importances") if result else None,
                feature_names=st.session_state.dataset_info.get("feature_names") if st.session_state.dataset_info else None
            )
    
    with tab3:
        render_experiment_history(st.session_state.experiment_history)
        st.divider()
        render_accuracy_trend(st.session_state.experiment_history)
    
    with tab4:
        render_classification_report(
            classification_report=result.get("classification_report") if result else None
        )


if __name__ == "__main__":
    main()
