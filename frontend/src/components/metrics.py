"""Metrics display components."""
from typing import Optional, Dict, Any
import streamlit as st


def render_metrics_cards(
    accuracy: Optional[float] = None,
    f1_score: Optional[float] = None,
    model_name: Optional[str] = None,
    dataset_info: Optional[Dict[str, Any]] = None
) -> None:
    """Render metric cards in a grid layout.
    
    Args:
        accuracy: Model accuracy score
        f1_score: Model F1 score
        model_name: Name of the trained model
        dataset_info: Dataset information including name, n_samples, n_features
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_accuracy_card(accuracy)
    
    with col2:
        render_f1_score_card(f1_score)
    
    with col3:
        render_model_card(model_name)
    
    with col4:
        render_dataset_card(dataset_info)


def render_accuracy_card(accuracy: Optional[float] = None) -> None:
    """Render accuracy metric card.
    
    Args:
        accuracy: Accuracy value (0-1 range)
    """
    st.metric(
        label="🎯 Accuracy",
        value=f"{accuracy:.4f}" if accuracy is not None else "N/A",
        help="Classification accuracy on test set"
    )


def render_f1_score_card(f1_score: Optional[float] = None) -> None:
    """Render F1 score metric card.
    
    Args:
        f1_score: F1 score value (0-1 range)
    """
    st.metric(
        label="📊 F1 Score",
        value=f"{f1_score:.4f}" if f1_score is not None else "N/A",
        help="Weighted F1 score on test set"
    )


def render_model_card(model_name: Optional[str] = None) -> None:
    """Render model name card.
    
    Args:
        model_name: Name of the trained model
    """
    display_name = model_name.replace("_", " ").title() if model_name else "N/A"
    st.metric(
        label="🤖 Model",
        value=display_name,
        help="Type of trained model"
    )


def render_dataset_card(dataset_info: Optional[Dict[str, Any]] = None) -> None:
    """Render dataset information card.
    
    Args:
        dataset_info: Dictionary containing dataset_name, n_samples, n_features
    """
    if dataset_info:
        dataset_name = dataset_info.get("dataset_name", "N/A")
        n_samples = dataset_info.get("n_samples", 0)
        n_features = dataset_info.get("n_features", 0)
        
        value_text = f"{dataset_name.title()}"
        delta_text = f"{n_samples} samples, {n_features} features"
        
        st.metric(
            label="📁 Dataset",
            value=value_text,
            delta=delta_text,
            help="Dataset information"
        )
    else:
        st.metric(
            label="📁 Dataset",
            value="N/A",
            help="Dataset information"
        )
